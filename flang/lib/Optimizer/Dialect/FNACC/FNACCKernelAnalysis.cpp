#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelPlan.h"

#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cassert>
#include <utility>

using namespace mlir;

namespace fir::fnacc {
namespace {

static ElementwiseRecognitionResult fail(fir::fnacc::LaunchOp launchOp,
                                         StringRef reason) {
  return ElementwiseRecognitionResult::failure(launchOp.getOperation(),
                                               reason.str());
}

static ElementwiseRecognitionResult fail(Operation *op, StringRef reason) {
  return ElementwiseRecognitionResult::failure(op, reason.str());
}

static ElementwiseRecognitionResult
failWithDefault(Operation *op, StringRef reason, StringRef fallback) {
  if (!reason.empty())
    return fail(op, reason);

  return fail(op, fallback);
}

static Value stripFirConvert(Value v) {
  while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
    v = cvt.getValue();

  return v;
}

/// Return a stable identity for an array base.
///
/// A single allocatable or assumed-shape array can be materialized multiple
/// times inside a launch as distinct fir.load/fir.box_addr chains. Those SSA
/// values are different, but they all describe the same runtime array when
/// they originate from the same descriptor.
static Value getArrayIdentity(Value value) {
  value = stripFirConvert(value);

  auto boxAddr = value.getDefiningOp<fir::BoxAddrOp>();
  if (!boxAddr)
    return value;

  Value box = stripFirConvert(boxAddr->getOperand(0));
  if (auto load = box.getDefiningOp<fir::LoadOp>())
    return stripFirConvert(load.getMemref());

  return box;
}

static bool sameArrayBase(Value lhs, Value rhs) {
  return lhs && rhs && getArrayIdentity(lhs) == getArrayIdentity(rhs);
}

static void markConsumed(ElementwiseKernel &kernel, Operation *op) {
  if (!op)
    return;

  if (!llvm::is_contained(kernel.consumedOps, op))
    kernel.consumedOps.push_back(op);
}

/// Mark a value's launch-local backward slice.
///
/// This follows only operands defined inside fnacc.launch. Values defined
/// outside the launch are captures and must remain available to host lowering;
/// their defining operations are not part of the region being erased.
static void markLaunchLocalBackwardSlice(ElementwiseKernel &kernel,
                                         fir::fnacc::LaunchOp launchOp,
                                         Value value) {
  Operation *definingOp = value.getDefiningOp();

  if (!definingOp || !launchOp->isProperAncestor(definingOp))
    return;

  if (llvm::is_contained(kernel.consumedOps, definingOp))
    return;

  markConsumed(kernel, definingOp);

  for (Value operand : definingOp->getOperands())
    markLaunchLocalBackwardSlice(kernel, launchOp, operand);
}

static void markPostLoopInductionUpdate(ElementwiseKernel &kernel,
                                        fir::fnacc::LaunchOp launchOp,
                                        fir::DoLoopOp loop,
                                        Value inductionMemref) {
  Block *parentBlock = loop->getBlock();
  bool afterLoop = false;

  for (Operation &op : parentBlock->getOperations()) {
    if (&op == loop.getOperation()) {
      afterLoop = true;
      continue;
    }

    if (!afterLoop)
      continue;

    auto store = dyn_cast<fir::StoreOp>(op);
    if (!store || store.getMemref() != inductionMemref)
      continue;

    markConsumed(kernel, store.getOperation());
    markLaunchLocalBackwardSlice(kernel, launchOp, store.getValue());
    return;
  }
}

/// Return a runtime-visible array base for a reduction.
///
/// Flang commonly lowers an allocatable array access inside fnacc.launch as:
///
///   %box = fir.load %descriptor
///   %addr = fir.box_addr %box
///   %element = fir.array_coor %addr ...
///
/// `%addr` is local to the launch region and becomes invalid when runtime
/// lowering erases that region.  The descriptor reference is the stable host
/// value from which runtime lowering can rematerialize the load and box_addr.
static std::optional<Value>
getRuntimeVisibleReductionArrayBase(fir::fnacc::LaunchOp launchOp,
                                    Value arrayBase) {
  Value original = stripFirConvert(arrayBase);
  Operation *originalDef = original.getDefiningOp();

  if (!originalDef || !launchOp->isProperAncestor(originalDef))
    return original;

  auto boxAddr = original.getDefiningOp<fir::BoxAddrOp>();
  if (!boxAddr)
    return std::nullopt;

  Value box = stripFirConvert(boxAddr->getOperand(0));
  auto load = box.getDefiningOp<fir::LoadOp>();
  if (!load)
    return std::nullopt;

  Value descriptorRef = stripFirConvert(load.getMemref());
  auto refTy = dyn_cast<fir::ReferenceType>(descriptorRef.getType());
  if (!refTy || !isa<fir::BoxType>(refTy.getEleTy()))
    return std::nullopt;

  Operation *descriptorDef = descriptorRef.getDefiningOp();
  if (!descriptorDef || launchOp->isProperAncestor(descriptorDef))
    return std::nullopt;

  return descriptorRef;
}

static std::optional<Value> getIntegerRefFromLoadLike(Value v) {
  while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
    v = cvt.getValue();

  auto load = v.getDefiningOp<fir::LoadOp>();
  if (!load)
    return std::nullopt;

  Value memref = load.getMemref();

  auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
  if (!refTy)
    return std::nullopt;

  Type elementType = refTy.getEleTy();
  if (!elementType.isInteger(8) && !elementType.isInteger(16) &&
      !elementType.isInteger(32) && !elementType.isInteger(64))
    return std::nullopt;

  return memref;
}

static std::optional<fir::fnacc::ElementwiseExtentSource>
getBoxDimExtentSource(Value v) {
  while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
    v = cvt.getValue();

  auto result = dyn_cast<OpResult>(v);
  if (!result)
    return std::nullopt;

  // fir.box_dims returns:
  //   result #0 = lower bound
  //   result #1 = extent
  //   result #2 = stride
  if (result.getResultNumber() != 1)
    return std::nullopt;

  auto boxDims = dyn_cast<fir::BoxDimsOp>(result.getOwner());
  if (!boxDims)
    return std::nullopt;

  fir::fnacc::ElementwiseExtentSource source;
  source.kind = fir::fnacc::ElementwiseExtentSourceKind::BoxDim;

  // In current FIR ODS this is commonly getVal(). If your generated accessor
  // differs, check FIROps.h.inc / FIROps.cpp.inc for fir.box_dims.
  source.value = boxDims.getVal();

  // Most generated FIR accessors call the dimension operand getDim().
  // If this does not compile in your tree, inspect the generated accessor name.
  if (auto cst =
          boxDims.getDim().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    source.dim = static_cast<unsigned>(cst.value());
  } else {
    // For now support only constant dimensions. The FIR in your assumed-shape
    // dump uses %c0, so this is enough for rank-1 assumed-shape arrays.
    return std::nullopt;
  }

  return source;
}

static fir::fnacc::ElementwiseExtentSource
getLoopExtentSource(fir::DoLoopOp loop) {
  Value ub = loop.getUpperBound();

  if (auto ref = getIntegerRefFromLoadLike(ub)) {
    fir::fnacc::ElementwiseExtentSource source;
    source.kind = fir::fnacc::ElementwiseExtentSourceKind::LoadIntegerRef;
    source.value = *ref;
    return source;
  }

  if (auto boxDim = getBoxDimExtentSource(ub))
    return *boxDim;

  fir::fnacc::ElementwiseExtentSource source;
  source.kind = fir::fnacc::ElementwiseExtentSourceKind::Value;
  source.value = ub;
  return source;
}

static bool isAcceptedStructuralOperation(Operation *op) {
  return isa<fir::ResultOp, fir::fnacc::TerminatorOp, arith::ConstantOp,
             arith::ConstantIndexOp, fir::ShapeOp, fir::ShapeShiftOp>(op);
}

static bool sameValueAfterFirConvert(Value lhs, Value rhs) {
  return lhs && rhs && stripFirConvert(lhs) == stripFirConvert(rhs);
}

static bool isRecognizedKernelStore(const ElementwiseKernel &kernel,
                                    fir::StoreOp store) {
  if (llvm::is_contained(kernel.consumedOps, store.getOperation()))
    return true;

  Value memref = stripFirConvert(store.getMemref());

  if (sameValueAfterFirConvert(memref, kernel.innerIndMemref) ||
      sameValueAfterFirConvert(memref, kernel.outerIndMemref) ||
      sameValueAfterFirConvert(memref, kernel.reductionIndMemref) ||
      sameValueAfterFirConvert(memref, kernel.accumulatorMemref) ||
      sameValueAfterFirConvert(memref, kernel.reductionScalarRef))
    return true;

  auto arrayCoor = memref.getDefiningOp<fir::ArrayCoorOp>();
  return arrayCoor && sameArrayBase(arrayCoor.getMemref(), kernel.writeArray);
}

static Operation *findDiscardedSideEffect(fir::fnacc::LaunchOp launchOp,
                                          const ElementwiseKernel &kernel) {
  Operation *unsupported = nullptr;

  launchOp.walk([&](Operation *op) {
    if (unsupported || op == launchOp.getOperation() ||
        llvm::is_contained(kernel.consumedOps, op) ||
        isAcceptedStructuralOperation(op))
      return;

    if (auto store = dyn_cast<fir::StoreOp>(op)) {
      if (!isRecognizedKernelStore(kernel, store))
        unsupported = op;
      return;
    }

    // Loads and operations with no memory effects are safe parts of a
    // recognized expression. Any other unaccounted operation would be erased
    // by host lowering, so reject the launch instead of silently discarding
    // its effects.
    if (isa<fir::LoadOp>(op) || isMemoryEffectFree(op))
      return;

    unsupported = op;
  });

  return unsupported;
}

static ElementwiseRecognitionResult
validateRecognizedKernel(fir::fnacc::LaunchOp launchOp,
                         ElementwiseRecognitionResult result) {
  ElementwiseKernel kernel = std::move(result.getKernel());

  if (Operation *unsupported = findDiscardedSideEffect(launchOp, kernel)) {
    std::string reason = "unsupported operation would be discarded: ";
    reason += unsupported->getName().getStringRef().str();
    return fail(unsupported, reason);
  }

  return ElementwiseRecognitionResult::success(std::move(kernel));
}

/// Return true if `v` is an integer constant equal to `expected`.
///
/// This handles values such as:
///
///   %c1_i32 = arith.constant 1 : i32
///
/// and also:
///
///   %idx = fir.convert %c1_i32 : (i32) -> index
static bool isConstantIntegerValue(Value v, int64_t expected) {
  v = stripFirConvert(v);

  llvm::APInt intValue;
  if (!matchPattern(v, m_ConstantInt(&intValue)))
    return false;

  if (intValue.getBitWidth() > 64)
    return false;

  return intValue.getSExtValue() == expected;
}

/// For now FNACC only supports canonical Fortran loops:
///
///   do i = 1, n
///
/// That corresponds to:
///
///   lower bound = constant 1
///   step        = constant 1
///
/// The generated Triton coordinates are zero-based offsets, so supporting
/// arbitrary lower bounds requires additional offset correction.
static bool verifyLoopLowerBoundAndStep(fir::DoLoopOp loop, StringRef loopName,
                                        std::string &reason) {
  if (!isConstantIntegerValue(loop.getLowerBound(), 1)) {
    reason = loopName.str() + " loop lower bound must be constant 1";
    return false;
  }

  if (!isConstantIntegerValue(loop.getStep(), 1)) {
    reason = loopName.str() + " loop step must be constant 1";
    return false;
  }

  return true;
}

/// Find the Fortran induction variable storage.
///
/// Flang often lowers:
///
///   fir.do_loop %arg4 = %lb to %ub step %step : i32 {
///     fir.store %arg4 to %i : !fir.ref<i32>
///     ...
///   }
///
/// The induction variable is the first block argument of the body.
static Value findInductionMemref(fir::DoLoopOp loop) {
  Block *body = loop.getBody();
  if (!body)
    return {};

  if (body->getNumArguments() == 0)
    return {};

  Value inductionVar = body->getArgument(0);

  for (Operation &op : body->getOperations()) {
    if (auto st = dyn_cast<fir::StoreOp>(op)) {
      Value stored = st.getValue();

      while (auto cvt = stored.getDefiningOp<fir::ConvertOp>())
        stored = cvt.getValue();

      if (stored == inductionVar)
        return st.getMemref();
    }
  }

  return {};
}

/// Trace an array_coor index through converts and verify that it is:
///
///   fir.convert(fir.load(%expectedMemref))
///
/// or:
///
///   fir.load(%expectedMemref)
static bool indexIsLoadOf(Value v, Value expectedMemref) {
  while (true) {
    if (auto cvt = v.getDefiningOp<fir::ConvertOp>()) {
      v = cvt.getValue();
      continue;
    }

    if (auto load = v.getDefiningOp<fir::LoadOp>())
      return load.getMemref() == expectedMemref;

    return false;
  }
}

static fir::fnacc::ElementType getSupportedElementType(Type type) {
  if (type.isInteger(8))
    return fir::fnacc::ElementType::I8;
  if (type.isInteger(16))
    return fir::fnacc::ElementType::I16;
  if (type.isInteger(32))
    return fir::fnacc::ElementType::I32;
  if (type.isInteger(64))
    return fir::fnacc::ElementType::I64;
  if (type.isF32())
    return fir::fnacc::ElementType::F32;
  if (type.isF64())
    return fir::fnacc::ElementType::F64;
  return fir::fnacc::ElementType::Unknown;
}

static std::optional<Value> getScalarElementRefFromValue(Value v) {
  v = stripFirConvert(v);

  auto load = v.getDefiningOp<fir::LoadOp>();
  if (!load)
    return std::nullopt;

  Value memref = load.getMemref();

  // If this is a load from an array_coor, it is an array element load, not
  // a scalar capture.
  if (memref.getDefiningOp<fir::ArrayCoorOp>())
    return std::nullopt;

  auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
  if (!refTy)
    return std::nullopt;

  Type eleTy = refTy.getEleTy();

  if (getSupportedElementType(eleTy) == fir::fnacc::ElementType::Unknown)
    return std::nullopt;

  return memref;
}

static fir::fnacc::ElementType getScalarRefElementType(Value memref) {
  auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
  if (!refTy)
    return fir::fnacc::ElementType::Unknown;

  return getSupportedElementType(refTy.getEleTy());
}

static Type unwrapArrayStorageType(Type type) {
  while (true) {
    if (auto refTy = dyn_cast<fir::ReferenceType>(type)) {
      type = refTy.getEleTy();
      continue;
    }

    if (auto boxTy = dyn_cast<fir::BoxType>(type)) {
      type = boxTy.getEleTy();
      continue;
    }

    if (auto heapTy = dyn_cast<fir::HeapType>(type)) {
      type = heapTy.getEleTy();
      continue;
    }

    if (auto ptrTy = dyn_cast<fir::PointerType>(type)) {
      type = ptrTy.getEleTy();
      continue;
    }

    return type;
  }
}

static fir::fnacc::ElementType getArrayElementType(Value v) {
  Type type = unwrapArrayStorageType(v.getType());
  auto arrTy = dyn_cast<fir::SequenceType>(type);

  if (!arrTy)
    return fir::fnacc::ElementType::Unknown;

  return getSupportedElementType(arrTy.getEleTy());
}

static bool inferAndCheckElementType(ElementwiseKernel &k,
                                     std::string &reason) {
  if (!k.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  fir::fnacc::ElementType type = getArrayElementType(k.writeArray);
  if (type == fir::fnacc::ElementType::Unknown) {
    reason = "write array must be integer(1/2/4/8) or real(4/8)";
    return false;
  }

  for (Value read : k.readArrays) {
    fir::fnacc::ElementType readType = getArrayElementType(read);
    if (readType != type) {
      reason = "all read/write arrays must have the same element type";
      return false;
    }
  }

  for (Value scalar : k.scalarRefs) {
    if (getScalarRefElementType(scalar) != type) {
      reason = "scalar captures must match array element type";
      return false;
    }
  }

  k.elementType = type;
  return true;
}

struct ArrayAccessInfo {
  llvm::SmallVector<Value> readArrays;
  llvm::SmallVector<Value> readValues;

  Value writeArray;
  Value storedValue;
};

static int findReadValueIndex(ArrayRef<Value> readValues, Value v) {
  v = stripFirConvert(v);

  for (auto it : llvm::enumerate(readValues)) {
    Value readValue = stripFirConvert(it.value());
    if (readValue == v)
      return static_cast<int>(it.index());
  }

  return -1;
}

static std::unique_ptr<ElementwiseExpr> makeExpr(ElementwiseExprKind kind) {
  auto expr = std::make_unique<ElementwiseExpr>();
  expr->kind = kind;
  return expr;
}

static bool isZeroReal(Value v) {
  v = stripFirConvert(v);

  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;

  auto floatAttr = dyn_cast<FloatAttr>(cst.getValue());
  if (!floatAttr)
    return false;

  if (!floatAttr.getType().isF32() && !floatAttr.getType().isF64())
    return false;

  return floatAttr.getValue().isZero();
}

static unsigned getOrAddValueIndex(llvm::SmallVectorImpl<Value> &values,
                                   Value value) {
  for (auto it : llvm::enumerate(values)) {
    if (it.value() == value)
      return it.index();
  }

  values.push_back(value);
  return values.size() - 1;
}

static unsigned getOrAddArrayIndex(llvm::SmallVectorImpl<Value> &arrays,
                                   Value array) {
  for (auto it : llvm::enumerate(arrays)) {
    if (sameArrayBase(it.value(), array))
      return it.index();
  }

  arrays.push_back(array);
  return arrays.size() - 1;
}

static std::optional<double> getRealConstantValue(Value v) {
  v = stripFirConvert(v);

  auto constant = v.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;

  auto floatAttr = dyn_cast<FloatAttr>(constant.getValue());
  if (!floatAttr)
    return std::nullopt;

  if (!floatAttr.getType().isF32() && !floatAttr.getType().isF64())
    return std::nullopt;

  return floatAttr.getValueAsDouble();
}

static std::optional<int64_t> getIntegerConstantValue(Value v) {
  v = stripFirConvert(v);

  auto constant = v.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;

  auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integerAttr || integerAttr.getType().isInteger(1) ||
      getSupportedElementType(integerAttr.getType()) == ElementType::Unknown)
    return std::nullopt;

  return integerAttr.getValue().getSExtValue();
}

/// Strip FIR operations that preserve an elementwise expression's value.
/// `fir.no_reassoc` records an optimization constraint, but it is not a
/// computation that needs to be reproduced in TTIR. The expression tree still
/// preserves the source evaluation order after this wrapper is removed.
static Value stripElementwiseWrappers(Value value) {
  while (true) {
    value = stripFirConvert(value);

    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getNumOperands() != 1 ||
        operation->getName().getStringRef() != "fir.no_reassoc")
      return value;

    value = operation->getOperand(0);
  }
}

/// Match the branch-free signed ABS sequence emitted by Flang:
///
///   sign = x >> (bitwidth - 1)
///   abs  = (x xor sign) - sign
static std::optional<Value> matchFlangIntegerAbs(Value value) {
  auto subtract = value.getDefiningOp<arith::SubIOp>();
  if (!subtract)
    return std::nullopt;

  Value xorValue = stripFirConvert(subtract.getLhs());
  Value signValue = stripFirConvert(subtract.getRhs());
  Operation *xorOperation = xorValue.getDefiningOp();
  Operation *shiftOperation = signValue.getDefiningOp();

  if (!xorOperation || xorOperation->getName().getStringRef() != "arith.xori" ||
      xorOperation->getNumOperands() != 2 || !shiftOperation ||
      shiftOperation->getName().getStringRef() != "arith.shrsi" ||
      shiftOperation->getNumOperands() != 2)
    return std::nullopt;

  Value original = stripFirConvert(shiftOperation->getOperand(0));
  auto integerType = dyn_cast<mlir::IntegerType>(original.getType());
  if (!integerType ||
      getSupportedElementType(integerType) == ElementType::Unknown)
    return std::nullopt;

  std::optional<int64_t> shiftAmount =
      getIntegerConstantValue(shiftOperation->getOperand(1));
  int64_t expectedShift = static_cast<int64_t>(integerType.getWidth()) - 1;
  if (!shiftAmount || *shiftAmount != expectedShift)
    return std::nullopt;

  Value xorLhs = stripFirConvert(xorOperation->getOperand(0));
  Value xorRhs = stripFirConvert(xorOperation->getOperand(1));
  bool matches = (xorLhs == original && xorRhs == signValue) ||
                 (xorRhs == original && xorLhs == signValue);
  if (!matches)
    return std::nullopt;

  return original;
}

static bool isSupportedElementType(Type type, ElementType expected) {
  ElementType actual = getSupportedElementType(type);
  if (expected == ElementType::Unknown)
    return actual != ElementType::Unknown;
  return actual == expected;
}

static std::unique_ptr<ElementwiseExpr>
recognizeElementwiseExpr(Value value, const ArrayAccessInfo &accesses,
                         ElementwiseKernel &kernel, std::string &reason);

static std::unique_ptr<ElementwiseExpr>
recognizeUnaryElementwiseExpr(ElementwiseExprKind kind, Value operand,
                              const ArrayAccessInfo &accesses,
                              ElementwiseKernel &kernel, std::string &reason) {
  auto child = recognizeElementwiseExpr(operand, accesses, kernel, reason);
  if (!child)
    return nullptr;

  auto expression = makeExpr(kind);
  expression->operands.push_back(std::move(child));
  return expression;
}

static std::unique_ptr<ElementwiseExpr> recognizeBinaryElementwiseExpr(
    ElementwiseExprKind kind, Value lhsValue, Value rhsValue,
    const ArrayAccessInfo &accesses, ElementwiseKernel &kernel,
    std::string &reason,
    ElementwiseExprResultKind resultKind = ElementwiseExprResultKind::Element) {
  auto lhs = recognizeElementwiseExpr(lhsValue, accesses, kernel, reason);
  if (!lhs)
    return nullptr;

  auto rhs = recognizeElementwiseExpr(rhsValue, accesses, kernel, reason);
  if (!rhs)
    return nullptr;

  auto expression = makeExpr(kind);
  expression->resultKind = resultKind;
  expression->operands.push_back(std::move(lhs));
  expression->operands.push_back(std::move(rhs));
  return expression;
}

static std::optional<ElementwiseExprKind>
getComparisonExprKind(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::OLT:
    return ElementwiseExprKind::CmpOLT;
  case arith::CmpFPredicate::OLE:
    return ElementwiseExprKind::CmpOLE;
  case arith::CmpFPredicate::OGT:
    return ElementwiseExprKind::CmpOGT;
  case arith::CmpFPredicate::OGE:
    return ElementwiseExprKind::CmpOGE;
  case arith::CmpFPredicate::OEQ:
    return ElementwiseExprKind::CmpOEQ;
  case arith::CmpFPredicate::ONE:
    return ElementwiseExprKind::CmpONE;
  default:
    return std::nullopt;
  }
}

static std::optional<ElementwiseExprKind>
getComparisonExprKind(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::slt:
    return ElementwiseExprKind::CmpSLT;
  case arith::CmpIPredicate::sle:
    return ElementwiseExprKind::CmpSLE;
  case arith::CmpIPredicate::sgt:
    return ElementwiseExprKind::CmpSGT;
  case arith::CmpIPredicate::sge:
    return ElementwiseExprKind::CmpSGE;
  case arith::CmpIPredicate::eq:
    return ElementwiseExprKind::CmpIEQ;
  case arith::CmpIPredicate::ne:
    return ElementwiseExprKind::CmpINE;
  default:
    return std::nullopt;
  }
}

static std::unique_ptr<ElementwiseExpr>
recognizeElementwiseExpr(Value value, const ArrayAccessInfo &accesses,
                         ElementwiseKernel &kernel, std::string &reason) {
  value = stripElementwiseWrappers(value);

  // Array element load previously collected from a recognized array_coor.
  int readIndex = findReadValueIndex(accesses.readValues, value);
  if (readIndex >= 0) {
    Value arrayBase = accesses.readArrays[readIndex];
    unsigned arrayIndex = getOrAddArrayIndex(kernel.readArrays, arrayBase);

    auto expression = makeExpr(ElementwiseExprKind::ArrayLoad);
    expression->source = kernel.readArrays[arrayIndex];
    return expression;
  }

  // Fortran scalar integer or real variable captured by reference.
  if (auto scalarRef = getScalarElementRefFromValue(value)) {
    getOrAddValueIndex(kernel.scalarRefs, *scalarRef);

    auto expression = makeExpr(ElementwiseExprKind::ScalarLoad);
    expression->source = *scalarRef;
    return expression;
  }

  // Floating-point constant.
  if (auto constantValue = getRealConstantValue(value)) {
    auto expression = makeExpr(ElementwiseExprKind::ConstantReal);
    expression->realValue = *constantValue;
    return expression;
  }

  if (auto constantValue = getIntegerConstantValue(value)) {
    auto expression = makeExpr(ElementwiseExprKind::ConstantInteger);
    expression->integerValue = *constantValue;
    return expression;
  }

  Operation *operation = value.getDefiningOp();
  if (!operation) {
    reason = "elementwise expression value has no defining operation";
    return nullptr;
  }

  Type resultType = value.getType();

  // Element results must use a supported integer or real type. i1 is accepted
  // only as the intermediate result of a comparison.
  if (!isSupportedElementType(resultType, ElementType::Unknown) &&
      !resultType.isInteger(1)) {
    reason = "elementwise expression has unsupported result type";
    return nullptr;
  }

  // Unary negation in canonical arith.negf form.
  if (auto neg = dyn_cast<arith::NegFOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::NegF, neg.getOperand(), accesses, kernel, reason);
  }

  // Flang or canonicalization may represent negation as 0.0 - value.
  if (auto sub = dyn_cast<arith::SubFOp>(operation)) {
    if (isZeroReal(sub.getLhs())) {
      return recognizeUnaryElementwiseExpr(
          ElementwiseExprKind::NegF, sub.getRhs(), accesses, kernel, reason);
    }

    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::SubF,
                                          sub.getLhs(), sub.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto abs = dyn_cast<math::AbsFOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::AbsF, abs.getOperand(), accesses, kernel, reason);
  }

  if (auto sqrt = dyn_cast<math::SqrtOp>(operation)) {
    return recognizeUnaryElementwiseExpr(ElementwiseExprKind::SqrtF,
                                         sqrt.getOperand(), accesses, kernel,
                                         reason);
  }

  if (auto exp = dyn_cast<math::ExpOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::ExpF, exp.getOperand(), accesses, kernel, reason);
  }

  if (auto log = dyn_cast<math::LogOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::LogF, log.getOperand(), accesses, kernel, reason);
  }

  if (auto sin = dyn_cast<math::SinOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::SinF, sin.getOperand(), accesses, kernel, reason);
  }

  if (auto cos = dyn_cast<math::CosOp>(operation)) {
    return recognizeUnaryElementwiseExpr(
        ElementwiseExprKind::CosF, cos.getOperand(), accesses, kernel, reason);
  }

  if (auto tanh = dyn_cast<math::TanhOp>(operation)) {
    return recognizeUnaryElementwiseExpr(ElementwiseExprKind::TanhF,
                                         tanh.getOperand(), accesses, kernel,
                                         reason);
  }

  StringRef operationName = operation->getName().getStringRef();

  if (operationName == "math.absi") {
    if (operation->getNumOperands() != 1) {
      reason = "integer absolute-value operation is not unary";
      return nullptr;
    }
    return recognizeUnaryElementwiseExpr(ElementwiseExprKind::AbsI,
                                         operation->getOperand(0), accesses,
                                         kernel, reason);
  }

  if (auto add = dyn_cast<arith::AddFOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::AddF,
                                          add.getLhs(), add.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto mul = dyn_cast<arith::MulFOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::MulF,
                                          mul.getLhs(), mul.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto div = dyn_cast<arith::DivFOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::DivF,
                                          div.getLhs(), div.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto add = dyn_cast<arith::AddIOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::AddI,
                                          add.getLhs(), add.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto sub = dyn_cast<arith::SubIOp>(operation)) {
    if (std::optional<Value> absOperand = matchFlangIntegerAbs(value))
      return recognizeUnaryElementwiseExpr(
          ElementwiseExprKind::AbsI, *absOperand, accesses, kernel, reason);

    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::SubI,
                                          sub.getLhs(), sub.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto mul = dyn_cast<arith::MulIOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::MulI,
                                          mul.getLhs(), mul.getRhs(), accesses,
                                          kernel, reason);
  }

  if (auto div = dyn_cast<arith::DivSIOp>(operation)) {
    return recognizeBinaryElementwiseExpr(ElementwiseExprKind::DivSI,
                                          div.getLhs(), div.getRhs(), accesses,
                                          kernel, reason);
  }

  // Use operation names for min/max because exact generated C++ class names
  // differ across MLIR revisions.

  if (operationName == "arith.minimumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point minimum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MinF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
  }

  if (operationName == "arith.maximumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point maximum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MaxF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
  }

  // Do not collapse minnum/maxnum into minimum/maximum: their NaN behavior is
  // different, and changing the operation while emitting TTIR would be a
  // silent semantic change.
  if (operationName == "arith.minnumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point minnum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MinNumF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
  }

  if (operationName == "arith.maxnumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point maxnum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MaxNumF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
  }

  if (operationName == "arith.minsi" || operationName == "arith.maxsi") {
    if (operation->getNumOperands() != 2) {
      reason = "signed integer min/max operation is not binary";
      return nullptr;
    }

    ElementwiseExprKind kind = operationName == "arith.minsi"
                                   ? ElementwiseExprKind::MinSI
                                   : ElementwiseExprKind::MaxSI;
    return recognizeBinaryElementwiseExpr(kind, operation->getOperand(0),
                                          operation->getOperand(1), accesses,
                                          kernel, reason);
  }

  if (auto compare = dyn_cast<arith::CmpFOp>(operation)) {
    std::optional<ElementwiseExprKind> comparisonKind =
        getComparisonExprKind(compare.getPredicate());

    if (!comparisonKind) {
      reason = "unsupported floating-point comparison predicate";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        *comparisonKind, compare.getLhs(), compare.getRhs(), accesses, kernel,
        reason, ElementwiseExprResultKind::Predicate);
  }

  if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
    std::optional<ElementwiseExprKind> comparisonKind =
        getComparisonExprKind(compare.getPredicate());

    if (!comparisonKind) {
      reason = "unsupported integer comparison predicate";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        *comparisonKind, compare.getLhs(), compare.getRhs(), accesses, kernel,
        reason, ElementwiseExprResultKind::Predicate);
  }

  if (auto select = dyn_cast<arith::SelectOp>(operation)) {
    auto condition = recognizeElementwiseExpr(select.getCondition(), accesses,
                                              kernel, reason);
    if (!condition)
      return nullptr;

    if (condition->resultKind != ElementwiseExprResultKind::Predicate) {
      reason = "elementwise select condition is not a predicate";
      return nullptr;
    }

    auto trueValue = recognizeElementwiseExpr(select.getTrueValue(), accesses,
                                              kernel, reason);
    if (!trueValue)
      return nullptr;

    if (trueValue->resultKind != ElementwiseExprResultKind::Element) {
      reason = "elementwise select true value is not an element value";
      return nullptr;
    }

    auto falseValue = recognizeElementwiseExpr(select.getFalseValue(), accesses,
                                               kernel, reason);
    if (!falseValue)
      return nullptr;

    if (falseValue->resultKind != ElementwiseExprResultKind::Element) {
      reason = "elementwise select false value is not an element value";
      return nullptr;
    }

    auto expression = makeExpr(ElementwiseExprKind::Select);
    expression->resultKind = ElementwiseExprResultKind::Element;
    expression->operands.push_back(std::move(condition));
    expression->operands.push_back(std::move(trueValue));
    expression->operands.push_back(std::move(falseValue));
    return expression;
  }

  reason = "unsupported operation in elementwise expression tree: ";
  reason += operationName;
  return nullptr;
}

static bool detectGenericExpr(ElementwiseKernel &k, const ArrayAccessInfo &info,
                              int rank, std::string &reason) {
  k.kind =
      rank == 1 ? ElementwiseKernelKind::Expr1D : ElementwiseKernelKind::Expr2D;
  k.rank = rank;
  k.writeArray = info.writeArray;
  k.readArrays.clear();
  k.scalarRefs.clear();
  k.computeOp = info.storedValue.getDefiningOp();

  auto expr = recognizeElementwiseExpr(info.storedValue, info, k, reason);
  if (!expr)
    return false;

  if (k.readArrays.size() > 3) {
    reason = "expression tree supports at most three read arrays";
    return false;
  }

  if (k.scalarRefs.size() > 3) {
    reason = "expression tree supports at most three scalar values";
    return false;
  }

  k.expression = std::move(expr);

  if (!inferAndCheckElementType(k, reason))
    return false;

  return true;
}

static bool collectArrayAccessesFromBody(Block *body, unsigned expectedRank,
                                         ArrayRef<Value> expectedIndexMemrefs,
                                         ArrayAccessInfo &info,
                                         std::string &reason,
                                         unsigned minReads = 1,
                                         unsigned maxReads = 3) {
  if (!body) {
    reason = "loop has no body";
    return false;
  }

  for (Operation &op : body->getOperations()) {
    auto ac = dyn_cast<fir::ArrayCoorOp>(op);
    if (!ac)
      continue;

    auto indices = ac.getIndices();
    if (indices.size() != expectedRank) {
      reason = "array_coor has unexpected rank";
      return false;
    }

    for (unsigned i = 0; i < expectedRank; ++i) {
      if (!indexIsLoadOf(indices[i], expectedIndexMemrefs[i])) {
        reason = "array index is not a load of the expected induction variable";
        return false;
      }
    }

    Value base = ac.getMemref();

    for (Operation *user : ac.getResult().getUsers()) {
      if (auto load = dyn_cast<fir::LoadOp>(user)) {
        info.readArrays.push_back(base);
        info.readValues.push_back(load.getResult());
      } else if (auto st = dyn_cast<fir::StoreOp>(user)) {
        if (info.writeArray) {
          reason = "kernel has more than one write array";
          return false;
        }

        info.writeArray = base;
        info.storedValue = st.getValue();
      } else {
        reason = "array_coor result has unsupported user";
        return false;
      }
    }
  }

  if (!info.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  llvm::SmallVector<Value> uniqueReadArrays;
  for (Value array : info.readArrays)
    getOrAddArrayIndex(uniqueReadArrays, array);

  if (uniqueReadArrays.size() < minReads ||
      uniqueReadArrays.size() > maxReads ||
      info.readArrays.size() != info.readValues.size()) {
    reason = "kernel has unsupported number of read arrays";
    return false;
  }

  return true;
}

static bool detectDirectBinaryArrayArray(ElementwiseKernel &k,
                                         const ArrayAccessInfo &info,
                                         std::string &reason) {
  Operation *op = info.storedValue.getDefiningOp();

  if (!isSupportedElementwiseCompute(op)) {
    reason = "stored value is not a supported binary elementwise op";
    return false;
  }

  if (op->getNumOperands() != 2) {
    reason = "compute op is not binary";
    return false;
  }

  int lhsIndex = findReadValueIndex(info.readValues, op->getOperand(0));
  int rhsIndex = findReadValueIndex(info.readValues, op->getOperand(1));

  if (lhsIndex < 0 || rhsIndex < 0 || lhsIndex == rhsIndex) {
    reason = "binary compute operands are not the two array loads";
    return false;
  }

  k.kind = ElementwiseKernelKind::BinaryArrayArray;
  k.readArrays = info.readArrays;
  k.writeArray = info.writeArray;
  k.computeOp = op;
  k.scalarRefs.clear();

  return true;
}

static bool detectSaxpy1D(ElementwiseKernel &k, const ArrayAccessInfo &info,
                          std::string &reason) {
  Operation *root = info.storedValue.getDefiningOp();

  if (!root || !isa<arith::AddFOp>(root)) {
    reason = "SAXPY pattern requires stored value to be arith.addf";
    return false;
  }

  if (root->getNumOperands() != 2) {
    reason = "SAXPY root add is not binary";
    return false;
  }

  auto tryMatch = [&](Value maybeMulValue, Value maybeAddArrayValue) -> bool {
    auto mul = maybeMulValue.getDefiningOp<arith::MulFOp>();
    if (!mul)
      return false;

    int addArrayIndex = findReadValueIndex(info.readValues, maybeAddArrayValue);
    if (addArrayIndex < 0)
      return false;

    Value mulLhs = mul.getLhs();
    Value mulRhs = mul.getRhs();

    int scaledArrayIndex = findReadValueIndex(info.readValues, mulLhs);
    std::optional<Value> scalarRef;

    if (scaledArrayIndex >= 0) {
      scalarRef = getScalarElementRefFromValue(mulRhs);
    } else {
      scaledArrayIndex = findReadValueIndex(info.readValues, mulRhs);
      if (scaledArrayIndex >= 0)
        scalarRef = getScalarElementRefFromValue(mulLhs);
    }

    if (scaledArrayIndex < 0 || !scalarRef)
      return false;

    if (scaledArrayIndex == addArrayIndex)
      return false;

    k.kind = ElementwiseKernelKind::Saxpy1D;
    k.rank = 1;

    k.readArrays.clear();
    k.readArrays.push_back(info.readArrays[scaledArrayIndex]); // scaled array
    k.readArrays.push_back(info.readArrays[addArrayIndex]);    // addend array

    k.writeArray = info.writeArray;
    k.computeOp = root;

    k.scalarRefs.clear();
    k.scalarRefs.push_back(*scalarRef);

    return true;
  };

  // Supports:
  //
  //   alpha * a + b
  //   b + alpha * a
  if (tryMatch(root->getOperand(0), root->getOperand(1)))
    return true;

  if (tryMatch(root->getOperand(1), root->getOperand(0)))
    return true;

  reason = "stored value is not a supported SAXPY pattern";
  return false;
}

static bool collectArrayAccesses1D(ElementwiseKernel &k, fir::DoLoopOp loop,
                                   Value indMemref, std::string &reason) {
  ArrayAccessInfo info;

  Block *body = loop.getBody();
  llvm::SmallVector<Value> indexMemrefs;
  indexMemrefs.push_back(indMemref);

  if (!collectArrayAccessesFromBody(body, 1, indexMemrefs, info, reason, 1, 3))
    return false;

  if (!info.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  if (info.readArrays.empty()) {
    reason = "kernel expected one to three read arrays";
    return false;
  }

  // Existing direct form:
  //
  //   c(i) = a(i) op b(i)
  std::string binaryReason;
  if (detectDirectBinaryArrayArray(k, info, binaryReason)) {
    if (!inferAndCheckElementType(k, reason))
      return false;

    return true;
  }

  // Existing restricted SAXPY:
  //
  //   c(i) = alpha * a(i) + b(i)
  std::string saxpyReason;
  if (detectSaxpy1D(k, info, saxpyReason)) {
    if (!inferAndCheckElementType(k, reason))
      return false;

    return true;
  }

  // Generic 1-D expression tree. Supports one to three read arrays and up to
  // three scalar captures.
  //
  //   c(i) = alpha * a(i) + beta * b(i)
  //   c(i) = (a(i) + b(i)) * alpha
  //
  // One-read expressions such as c(i) = a(i) + 1.0 require a future ABI update.
  std::string exprReason;
  if (detectGenericExpr(k, info, 1, exprReason))
    return true;

  reason = "unsupported 1-D elementwise expression; binary failure: ";

  reason += binaryReason;
  reason += "; SAXPY failure: ";
  reason += saxpyReason;
  reason += "; expression-tree failure: ";
  reason += exprReason;

  return false;
}

static bool collectArrayAccesses2D(ElementwiseKernel &k,
                                   fir::DoLoopOp innerLoop,
                                   Value innerIndMemref, Value outerIndMemref,
                                   std::string &reason) {
  ArrayAccessInfo info;

  llvm::SmallVector<Value> indexMemrefs;
  indexMemrefs.push_back(innerIndMemref);
  indexMemrefs.push_back(outerIndMemref);

  if (!collectArrayAccessesFromBody(innerLoop.getBody(), 2, indexMemrefs, info,
                                    reason, 1, 3))
    return false;

  // Existing direct binary 2-D form:
  //
  //   c(i,j) = a(i,j) op b(i,j)
  std::string binaryReason;
  if (info.readArrays.size() == 2 &&
      detectDirectBinaryArrayArray(k, info, binaryReason)) {
    k.rank = 2;
    k.scalarRefs.clear();

    if (!inferAndCheckElementType(k, reason))
      return false;

    return true;
  }

  // New generic 2-D expression form:
  //
  //   c(i,j) = alpha * a(i,j) + beta * b(i,j)
  //   c(i,j) = a(i,j) + b(i,j) + d(i,j)
  //   c(i,j) = a(i,j) + 1.0
  std::string exprReason;
  if (detectGenericExpr(k, info, 2, exprReason))
    return true;

  reason = "unsupported 2-D expression; binary failure: ";
  reason += binaryReason;
  reason += "; expression-tree failure: ";
  reason += exprReason;

  return false;
}

static fir::fnacc::ElementType getRealRefElementType(Value v) {
  auto refTy = dyn_cast<fir::ReferenceType>(v.getType());
  if (!refTy)
    return fir::fnacc::ElementType::Unknown;

  Type eleTy = refTy.getEleTy();

  if (eleTy.isF32())
    return fir::fnacc::ElementType::F32;

  if (eleTy.isF64())
    return fir::fnacc::ElementType::F64;

  return fir::fnacc::ElementType::Unknown;
}

static bool isRealRef(Value v) {
  return getRealRefElementType(v) != fir::fnacc::ElementType::Unknown;
}

struct ReductionArrayLoadInfo {
  Value arrayBase;
  Value loadedValue;
};

static std::optional<Value>
getSingleReductionScalarFromLaunch(fir::fnacc::LaunchOp launchOp) {
  auto slotsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_slots");

  if (!slotsAttr)
    return std::nullopt;

  llvm::ArrayRef<int32_t> slots = slotsAttr.asArrayRef();

  if (slots.size() != 1)
    return std::nullopt;

  int32_t slot = slots[0];

  if (slot < 0)
    return std::nullopt;

  auto packVars = launchOp.getPackVars();

  if (static_cast<unsigned>(slot) >= packVars.size())
    return std::nullopt;

  return packVars[slot];
}

static std::optional<ReductionOperator>
getSingleReductionOperator(fir::fnacc::LaunchOp launchOp) {
  auto opsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_ops");

  if (!opsAttr)
    return ReductionOperator::Add; // Backwards-compatible default.

  llvm::ArrayRef<int32_t> ops = opsAttr.asArrayRef();

  if (ops.empty())
    return ReductionOperator::Add;

  if (ops.size() != 1)
    return std::nullopt;

  switch (ops.front()) {
  case 0:
    return ReductionOperator::Add;
  case 1:
    return ReductionOperator::Multiply;
  case 2:
    return ReductionOperator::Min;
  case 3:
    return ReductionOperator::Max;
  default:
    return std::nullopt;
  }
}

static bool collectReductionArrayLoads1D(
    fir::fnacc::LaunchOp launchOp, fir::DoLoopOp loop, Value indMemref,
    llvm::SmallVectorImpl<ReductionArrayLoadInfo> &loads, std::string &reason) {
  Block *body = loop.getBody();

  if (!body) {
    reason = "reduction loop has no body";
    return false;
  }

  for (Operation &op : body->getOperations()) {
    auto ac = dyn_cast<fir::ArrayCoorOp>(op);
    if (!ac)
      continue;

    auto indices = ac.getIndices();

    if (indices.size() != 1) {
      reason = "reduction only supports rank-1 array accesses";
      return false;
    }

    if (!indexIsLoadOf(indices[0], indMemref)) {
      reason =
          "reduction array index is not a load of the expected induction var";
      return false;
    }

    std::optional<Value> base =
        getRuntimeVisibleReductionArrayBase(launchOp, ac.getMemref());
    if (!base) {
      reason = "reduction array address cannot be materialized outside "
               "fnacc.launch";
      return false;
    }

    for (Operation *user : ac.getResult().getUsers()) {
      if (auto load = dyn_cast<fir::LoadOp>(user)) {
        ReductionArrayLoadInfo info;
        info.arrayBase = *base;
        info.loadedValue = load.getResult();
        loads.push_back(info);
      } else {
        reason = "reduction array_coor result has unsupported user";
        return false;
      }
    }
  }

  if (loads.empty()) {
    reason = "reduction has no array loads";
    return false;
  }

  if (loads.size() > 2) {
    reason = "reduction supports at most two array loads";
    return false;
  }

  return true;
}

static int findReductionLoadedValueIndex(ArrayRef<ReductionArrayLoadInfo> loads,
                                         Value value) {
  value = stripFirConvert(value);

  for (auto it : llvm::enumerate(loads)) {
    Value loaded = stripFirConvert(it.value().loadedValue);
    if (loaded == value)
      return static_cast<int>(it.index());
  }

  return -1;
}

static bool getOrCheckReductionElementType(ElementwiseKernel &k,
                                           std::string &reason) {
  if (k.readArrays.empty()) {
    reason = "reduction has no read arrays";
    return false;
  }

  fir::fnacc::ElementType type = getArrayElementType(k.readArrays[0]);

  if (type == fir::fnacc::ElementType::Unknown) {
    reason = "reduction read array must be integer(1/2/4/8) or real(4/8)";
    return false;
  }

  for (Value read : k.readArrays) {
    if (getArrayElementType(read) != type) {
      reason = "all reduction read arrays must have same element type";
      return false;
    }
  }

  if (!k.reductionScalarRef) {
    reason = "reduction has no scalar result reference";
    return false;
  }

  if (getScalarRefElementType(k.reductionScalarRef) != type) {
    reason = "reduction scalar type must match array element type";
    return false;
  }

  k.elementType = type;
  return true;
}

static bool valueIsLoadOfMemref(Value v, Value memref) {
  v = stripFirConvert(v);

  auto load = v.getDefiningOp<fir::LoadOp>();
  return load && load.getMemref() == memref;
}

static void markLoopBounds(ElementwiseKernel &kernel,
                           fir::fnacc::LaunchOp launchOp, fir::DoLoopOp loop) {
  markConsumed(kernel, loop.getOperation());

  markLaunchLocalBackwardSlice(kernel, launchOp, loop.getLowerBound());
  markLaunchLocalBackwardSlice(kernel, launchOp, loop.getUpperBound());
  markLaunchLocalBackwardSlice(kernel, launchOp, loop.getStep());
}

static void markInductionStore(ElementwiseKernel &kernel, fir::DoLoopOp loop,
                               Value inductionMemref) {
  Block *body = loop.getBody();
  if (!body)
    return;

  for (Operation &op : body->getOperations()) {
    auto store = dyn_cast<fir::StoreOp>(op);
    if (!store || store.getMemref() != inductionMemref)
      continue;

    Value stored = stripFirConvert(store.getValue());
    if (stored == body->getArgument(0)) {
      markConsumed(kernel, store.getOperation());
      return;
    }
  }
}

static bool reductionOperationMatches(Operation *op,
                                      ReductionOperator reductionOp) {
  if (!op)
    return false;

  StringRef name = op->getName().getStringRef();
  switch (reductionOp) {
  case ReductionOperator::Add:
    return name == "arith.addf" || name == "arith.addi";
  case ReductionOperator::Multiply:
    return name == "arith.mulf" || name == "arith.muli";
  case ReductionOperator::Min:
    if (name == "arith.minimumf" || name == "arith.minnumf" ||
        name == "arith.minsi")
      return true;
    break;
  case ReductionOperator::Max:
    if (name == "arith.maximumf" || name == "arith.maxnumf" ||
        name == "arith.maxsi")
      return true;
    break;
  }

  auto select = dyn_cast<arith::SelectOp>(op);
  if (!select || (reductionOp != ReductionOperator::Min &&
                  reductionOp != ReductionOperator::Max))
    return false;

  Value cmpLhs;
  Value cmpRhs;
  bool less = false;
  bool greater = false;

  if (auto compare = select.getCondition().getDefiningOp<arith::CmpFOp>()) {
    cmpLhs = stripFirConvert(compare.getLhs());
    cmpRhs = stripFirConvert(compare.getRhs());

    using Predicate = arith::CmpFPredicate;
    Predicate predicate = compare.getPredicate();
    less = predicate == Predicate::OLT || predicate == Predicate::OLE ||
           predicate == Predicate::ULT || predicate == Predicate::ULE;
    greater = predicate == Predicate::OGT || predicate == Predicate::OGE ||
              predicate == Predicate::UGT || predicate == Predicate::UGE;
  } else if (auto compare =
                 select.getCondition().getDefiningOp<arith::CmpIOp>()) {
    cmpLhs = stripFirConvert(compare.getLhs());
    cmpRhs = stripFirConvert(compare.getRhs());

    using Predicate = arith::CmpIPredicate;
    Predicate predicate = compare.getPredicate();
    less = predicate == Predicate::slt || predicate == Predicate::sle;
    greater = predicate == Predicate::sgt || predicate == Predicate::sge;
  } else {
    return false;
  }

  Value trueValue = stripFirConvert(select.getTrueValue());
  Value falseValue = stripFirConvert(select.getFalseValue());

  bool direct = trueValue == cmpLhs && falseValue == cmpRhs;
  bool reversed = trueValue == cmpRhs && falseValue == cmpLhs;
  if (!direct && !reversed)
    return false;

  if (reversed)
    std::swap(less, greater);

  return reductionOp == ReductionOperator::Min ? less : greater;
}

static bool matchSimpleReductionValue(Value value,
                                      ArrayRef<ReductionArrayLoadInfo> loads,
                                      Value reductionScalarRef,
                                      ReductionOperator reductionOp,
                                      Value &arrayBase, Operation *&computeOp,
                                      std::string &reason) {
  value = stripFirConvert(value);

  Operation *op = value.getDefiningOp();
  if (!reductionOperationMatches(op, reductionOp)) {
    reason = "reduction store value does not match directive operator";
    return false;
  }

  unsigned lhsIndex = op->getName().getStringRef() == "arith.select" ? 1 : 0;
  unsigned rhsIndex = lhsIndex + 1;
  if (op->getNumOperands() <= rhsIndex) {
    reason = "reduction operation has too few operands";
    return false;
  }

  Value lhs = stripFirConvert(op->getOperand(lhsIndex));
  Value rhs = stripFirConvert(op->getOperand(rhsIndex));

  Value candidate;

  if (valueIsLoadOfMemref(lhs, reductionScalarRef)) {
    candidate = rhs;
  } else if (valueIsLoadOfMemref(rhs, reductionScalarRef)) {
    candidate = lhs;
  } else {
    reason = "reduction operation does not include previous scalar value";
    return false;
  }

  int loadIndex = findReductionLoadedValueIndex(loads, candidate);
  if (loadIndex < 0) {
    reason = "reduction term is not an array element load";
    return false;
  }

  arrayBase = loads[loadIndex].arrayBase;
  computeOp = op;
  return true;
}

static bool matchReductionDotValue(Value value,
                                   ArrayRef<ReductionArrayLoadInfo> loads,
                                   Value reductionScalarRef, Value &arrayA,
                                   Value &arrayB, Operation *&computeOp,
                                   std::string &reason) {
  value = stripFirConvert(value);

  Operation *add = value.getDefiningOp();
  if (!add ||
      (add->getName().getStringRef() != "arith.addf" &&
       add->getName().getStringRef() != "arith.addi") ||
      add->getNumOperands() != 2) {
    reason = "dot reduction store value is not a supported add";
    return false;
  }

  Value lhs = stripFirConvert(add->getOperand(0));
  Value rhs = stripFirConvert(add->getOperand(1));

  Value maybeMul;

  if (valueIsLoadOfMemref(lhs, reductionScalarRef)) {
    maybeMul = rhs;
  } else if (valueIsLoadOfMemref(rhs, reductionScalarRef)) {
    maybeMul = lhs;
  } else {
    reason = "dot reduction add does not include previous scalar value";
    return false;
  }

  maybeMul = stripFirConvert(maybeMul);

  Operation *mul = maybeMul.getDefiningOp();
  if (!mul ||
      (mul->getName().getStringRef() != "arith.mulf" &&
       mul->getName().getStringRef() != "arith.muli") ||
      mul->getNumOperands() != 2) {
    reason = "dot reduction term is not a supported multiply";
    return false;
  }

  Value mulLhs = stripFirConvert(mul->getOperand(0));
  Value mulRhs = stripFirConvert(mul->getOperand(1));

  int lhsIndex = findReductionLoadedValueIndex(loads, mulLhs);
  int rhsIndex = findReductionLoadedValueIndex(loads, mulRhs);

  if (lhsIndex < 0 || rhsIndex < 0 || lhsIndex == rhsIndex) {
    reason = "dot reduction multiply operands are not two distinct array loads";
    return false;
  }

  arrayA = loads[lhsIndex].arrayBase;
  arrayB = loads[rhsIndex].arrayBase;
  computeOp = mul;
  return true;
}

static ElementwiseRecognitionResult
recognizeReduction1D(fir::fnacc::LaunchOp launchOp) {
  if (!launchOp->hasAttr("fnacc.reduction_slots"))
    return fail(launchOp, "launch has no FNACC reduction metadata");

  std::optional<ReductionOperator> reductionOp =
      getSingleReductionOperator(launchOp);
  if (!reductionOp)
    return fail(launchOp, "reduction requires one supported operator");

  std::optional<Value> reductionScalar =
      getSingleReductionScalarFromLaunch(launchOp);

  if (!reductionScalar)
    return fail(launchOp,
                "reduction recognition requires exactly one reduction scalar");

  Region &region = launchOp.getRegion();

  if (region.empty())
    return fail(launchOp, "reduction launch region is empty");

  Block &launchBlock = region.front();

  fir::DoLoopOp loop;
  for (Operation &op : launchBlock) {
    if (auto dl = dyn_cast<fir::DoLoopOp>(op)) {
      if (loop)
        return fail(
            &op, "reduction recognition expected exactly one top-level loop");
      loop = dl;
    }
  }

  if (!loop)
    return fail(launchOp, "reduction recognition found no top-level loop");

  std::string loopReason;
  if (!verifyLoopLowerBoundAndStep(loop, "reduction", loopReason))
    return fail(loop.getOperation(), loopReason);

  ElementwiseExtentSource extentX = getLoopExtentSource(loop);

  if (extentX.kind == ElementwiseExtentSourceKind::Unknown)
    return fail(loop.getOperation(), "could not determine reduction extent");

  Value indMemref = findInductionMemref(loop);
  if (!indMemref)
    return fail(loop.getOperation(),
                "could not find reduction induction variable storage");

  llvm::SmallVector<ReductionArrayLoadInfo> loads;
  std::string reason;

  if (!collectReductionArrayLoads1D(launchOp, loop, indMemref, loads, reason))
    return fail(loop.getOperation(), reason);

  fir::StoreOp reductionStore;

  for (Operation &op : loop.getBody()->getOperations()) {
    auto st = dyn_cast<fir::StoreOp>(op);
    if (!st)
      continue;

    if (st.getMemref() == *reductionScalar) {
      if (reductionStore)
        return fail(st.getOperation(),
                    "reduction loop has multiple stores to reduction scalar");
      reductionStore = st;
    }
  }

  if (!reductionStore)
    return fail(loop.getOperation(),
                "reduction loop does not store to reduction scalar");

  ElementwiseKernel k;
  k.rank = 1;
  k.loop1D = loop;
  k.extentX = extentX;
  k.innerIndMemref = indMemref;
  k.reductionScalarRef = *reductionScalar;
  k.reductionOperator = *reductionOp;
  k.writeArray = {};
  k.scalarRefs.clear();

  switch (extentX.kind) {
  case ElementwiseExtentSourceKind::Value:
    markLaunchLocalBackwardSlice(k, launchOp, extentX.value);
    break;

  case ElementwiseExtentSourceKind::LoadIntegerRef:
  case ElementwiseExtentSourceKind::BoxDim:
    // These sources intentionally retain a value defined outside the launch.
    break;

  case ElementwiseExtentSourceKind::Unknown:
    break;
  }

  markLoopBounds(k, launchOp, loop);
  markInductionStore(k, loop, indMemref);
  markPostLoopInductionUpdate(k, launchOp, loop, indMemref);
  markConsumed(k, reductionStore.getOperation());
  markLaunchLocalBackwardSlice(k, launchOp, reductionStore.getValue());

  // Try dot first because it is also an additive reduction.
  Value arrayA;
  Value arrayB;
  Operation *computeOp = nullptr;

  std::string dotReason;
  if (*reductionOp == ReductionOperator::Add &&
      matchReductionDotValue(reductionStore.getValue(), loads, *reductionScalar,
                             arrayA, arrayB, computeOp, dotReason)) {
    k.kind = ElementwiseKernelKind::ReductionDot1D;
    k.readArrays.push_back(arrayA);
    k.readArrays.push_back(arrayB);
    k.computeOp = computeOp;

    if (!getOrCheckReductionElementType(k, reason))
      return fail(loop.getOperation(), reason);

    return ElementwiseRecognitionResult::success(std::move(k));
  }

  Value arrayBase;
  computeOp = nullptr;

  std::string simpleReason;
  if (matchSimpleReductionValue(reductionStore.getValue(), loads,
                                *reductionScalar, *reductionOp, arrayBase,
                                computeOp, simpleReason)) {
    switch (*reductionOp) {
    case ReductionOperator::Add:
      k.kind = ElementwiseKernelKind::ReductionSum1D;
      break;
    case ReductionOperator::Multiply:
      k.kind = ElementwiseKernelKind::ReductionProduct1D;
      break;
    case ReductionOperator::Min:
      k.kind = ElementwiseKernelKind::ReductionMin1D;
      break;
    case ReductionOperator::Max:
      k.kind = ElementwiseKernelKind::ReductionMax1D;
      break;
    }
    k.readArrays.push_back(arrayBase);
    k.computeOp = computeOp;

    if (!getOrCheckReductionElementType(k, reason))
      return fail(loop.getOperation(), reason);

    return ElementwiseRecognitionResult::success(std::move(k));
  }

  reason = "unsupported reduction expression; dot failure: ";
  reason += dotReason;
  reason += "; simple reduction failure: ";
  reason += simpleReason;

  return fail(loop.getOperation(), reason);
}

static bool loadIsArrayAccess(Value v, Value expectedIndex0,
                              Value expectedIndex1, Value &arrayBase) {
  v = stripFirConvert(v);

  auto load = v.getDefiningOp<fir::LoadOp>();
  if (!load)
    return false;

  Value memref = load.getMemref();
  auto ac = memref.getDefiningOp<fir::ArrayCoorOp>();
  if (!ac)
    return false;

  auto indices = ac.getIndices();
  if (indices.size() != 2)
    return false;

  if (!indexIsLoadOf(indices[0], expectedIndex0))
    return false;

  if (!indexIsLoadOf(indices[1], expectedIndex1))
    return false;

  arrayBase = ac.getMemref();
  return true;
}

static bool findAccumulatorInit(fir::DoLoopOp iLoop, fir::DoLoopOp pLoop,
                                Value &accMemref, std::string &reason) {
  Block *body = iLoop.getBody();
  if (!body) {
    reason = "matmul i-loop has no body";
    return false;
  }

  for (Operation &op : body->getOperations()) {
    if (&op == pLoop.getOperation())
      break;

    auto st = dyn_cast<fir::StoreOp>(op);
    if (!st)
      continue;

    if (!isZeroReal(st.getValue()))
      continue;

    if (!isRealRef(st.getMemref()))
      continue;

    accMemref = st.getMemref();
    return true;
  }

  reason =
      "matmul did not find accumulator initialisation before reduction loop";
  return false;
}

static bool findMatmulReductionBody(fir::DoLoopOp pLoop, Value accMemref,
                                    Value iMemref, Value jMemref, Value pMemref,
                                    Value &aArray, Value &bArray,
                                    Operation *&computeOp,
                                    std::string &reason) {
  Block *body = pLoop.getBody();
  if (!body) {
    reason = "matmul reduction loop has no body";
    return false;
  }

  for (Operation &op : body->getOperations()) {
    auto st = dyn_cast<fir::StoreOp>(op);
    if (!st)
      continue;

    if (st.getMemref() != accMemref)
      continue;

    Value stored = stripFirConvert(st.getValue());
    Operation *addOp = stored.getDefiningOp();

    if (!addOp || !isa<arith::AddFOp>(addOp)) {
      reason = "matmul accumulator store is not arith.addf";
      return false;
    }

    if (addOp->getNumOperands() != 2) {
      reason = "matmul accumulator add is not binary";
      return false;
    }

    Value lhs = stripFirConvert(addOp->getOperand(0));
    Value rhs = stripFirConvert(addOp->getOperand(1));

    Value maybeMul;

    if (valueIsLoadOfMemref(lhs, accMemref)) {
      maybeMul = rhs;
    } else if (valueIsLoadOfMemref(rhs, accMemref)) {
      maybeMul = lhs;
    } else {
      reason = "matmul add does not include previous accumulator value";
      return false;
    }

    maybeMul = stripFirConvert(maybeMul);
    Operation *mulOp = maybeMul.getDefiningOp();

    if (!mulOp || !isa<arith::MulFOp>(mulOp)) {
      reason = "matmul accumulator update does not multiply two array loads";
      return false;
    }

    if (mulOp->getNumOperands() != 2) {
      reason = "matmul multiply is not binary";
      return false;
    }

    Value mulLhs = stripFirConvert(mulOp->getOperand(0));
    Value mulRhs = stripFirConvert(mulOp->getOperand(1));

    Value lhsArray;
    Value rhsArray;

    bool lhsIsA = loadIsArrayAccess(mulLhs, iMemref, pMemref, lhsArray);
    bool rhsIsB = loadIsArrayAccess(mulRhs, pMemref, jMemref, rhsArray);

    if (lhsIsA && rhsIsB) {
      aArray = lhsArray;
      bArray = rhsArray;
      computeOp = mulOp;
      return true;
    }

    bool rhsIsA = loadIsArrayAccess(mulRhs, iMemref, pMemref, rhsArray);
    bool lhsIsB = loadIsArrayAccess(mulLhs, pMemref, jMemref, lhsArray);

    if (rhsIsA && lhsIsB) {
      aArray = rhsArray;
      bArray = lhsArray;
      computeOp = mulOp;
      return true;
    }

    reason = "matmul multiply operands are not A(i,p) and B(p,j)";
    return false;
  }

  reason = "matmul reduction loop did not store updated accumulator";
  return false;
}

static bool findMatmulFinalStore(fir::DoLoopOp iLoop, fir::DoLoopOp pLoop,
                                 Value accMemref, Value iMemref, Value jMemref,
                                 Value &cArray, std::string &reason) {
  Block *body = iLoop.getBody();
  if (!body) {
    reason = "matmul i-loop has no body";
    return false;
  }

  bool afterReduction = false;

  for (Operation &op : body->getOperations()) {
    if (&op == pLoop.getOperation()) {
      afterReduction = true;
      continue;
    }

    if (!afterReduction)
      continue;

    auto st = dyn_cast<fir::StoreOp>(op);
    if (!st)
      continue;

    if (!valueIsLoadOfMemref(st.getValue(), accMemref)) {
      continue;
    }

    auto ac = st.getMemref().getDefiningOp<fir::ArrayCoorOp>();
    if (!ac)
      continue;

    auto indices = ac.getIndices();
    if (indices.size() != 2)
      continue;

    if (!indexIsLoadOf(indices[0], iMemref))
      continue;

    if (!indexIsLoadOf(indices[1], jMemref))
      continue;

    cArray = ac.getMemref();
    return true;
  }

  reason = "matmul did not find final C(i,j) = acc store";
  return false;
}

static ElementwiseRecognitionResult
recognizeMatMul2D(fir::fnacc::LaunchOp launchOp) {
  Region &region = launchOp.getRegion();
  if (region.empty())
    return fail(launchOp, "launch region is empty");

  Block &launchBlock = region.front();

  fir::DoLoopOp jLoop;
  for (Operation &op : launchBlock) {
    if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
      if (jLoop)
        return fail(&op, "matmul expected exactly one outer j loop");
      jLoop = loop;
    }
  }

  if (!jLoop)
    return fail(launchOp, "matmul found no outer j loop");

  Block *jBody = jLoop.getBody();
  if (!jBody)
    return fail(jLoop.getOperation(), "matmul j loop has no body");

  fir::DoLoopOp iLoop;
  for (Operation &op : jBody->getOperations()) {
    if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
      if (iLoop)
        return fail(&op, "matmul expected exactly one i loop inside j loop");
      iLoop = loop;
    }
  }

  if (!iLoop)
    return fail(jLoop.getOperation(), "matmul found no i loop");

  Block *iBody = iLoop.getBody();
  if (!iBody)
    return fail(iLoop.getOperation(), "matmul i loop has no body");

  fir::DoLoopOp pLoop;
  for (Operation &op : iBody->getOperations()) {
    if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
      if (pLoop)
        return fail(&op, "matmul expected exactly one reduction p loop");
      pLoop = loop;
    }
  }

  if (!pLoop)
    return fail(iLoop.getOperation(), "matmul found no reduction p loop");

  std::string loopReason;
  if (!verifyLoopLowerBoundAndStep(jLoop, "matmul j", loopReason))
    return fail(jLoop.getOperation(), loopReason);

  if (!verifyLoopLowerBoundAndStep(iLoop, "matmul i", loopReason))
    return fail(iLoop.getOperation(), loopReason);

  if (!verifyLoopLowerBoundAndStep(pLoop, "matmul p", loopReason))
    return fail(pLoop.getOperation(), loopReason);

  Value jMemref = findInductionMemref(jLoop);
  Value iMemref = findInductionMemref(iLoop);
  Value pMemref = findInductionMemref(pLoop);

  if (!jMemref)
    return fail(jLoop.getOperation(), "could not find j induction variable");

  if (!iMemref)
    return fail(iLoop.getOperation(), "could not find i induction variable");

  if (!pMemref)
    return fail(pLoop.getOperation(), "could not find p induction variable");

  Value accMemref;
  std::string reason;

  if (!findAccumulatorInit(iLoop, pLoop, accMemref, reason))
    return failWithDefault(iLoop.getOperation(), reason,
                           "matmul failed to find accumulator initialisation");

  Value aArray;
  Value bArray;
  Operation *computeOp = nullptr;

  if (!findMatmulReductionBody(pLoop, accMemref, iMemref, jMemref, pMemref,
                               aArray, bArray, computeOp, reason))
    return failWithDefault(pLoop.getOperation(), reason,
                           "matmul failed to match reduction loop body");

  Value cArray;

  if (!findMatmulFinalStore(iLoop, pLoop, accMemref, iMemref, jMemref, cArray,
                            reason))
    return failWithDefault(iLoop.getOperation(), reason,
                           "matmul failed to find final C(i,j) store");

  ElementwiseKernel k;
  k.kind = ElementwiseKernelKind::MatMul2D;
  k.rank = 2;

  k.outerLoop = jLoop;
  k.innerLoop = iLoop;
  k.reductionLoop = pLoop;

  k.outerIndMemref = jMemref;
  k.innerIndMemref = iMemref;
  k.reductionIndMemref = pMemref;
  k.accumulatorMemref = accMemref;

  k.extentY = getLoopExtentSource(jLoop); // m
  k.extentX = getLoopExtentSource(iLoop); // n
  k.extentZ = getLoopExtentSource(pLoop); // k

  k.readArrays.push_back(aArray);
  k.readArrays.push_back(bArray);
  k.writeArray = cArray;
  k.computeOp = computeOp;

  markLoopBounds(k, launchOp, jLoop);
  markLoopBounds(k, launchOp, iLoop);
  markLoopBounds(k, launchOp, pLoop);
  markInductionStore(k, iLoop, iMemref);
  markInductionStore(k, jLoop, jMemref);
  markInductionStore(k, pLoop, pMemref);

  if (!inferAndCheckElementType(k, reason))
    return failWithDefault(iLoop.getOperation(), reason,
                           "matmul failed element type inference");

  if (getRealRefElementType(accMemref) != k.elementType) {
    return fail(iLoop.getOperation(),
                "matmul accumulator type does not match array element type");
  }

  return ElementwiseRecognitionResult::success(std::move(k));
}

static ElementwiseRecognitionResult recognize1D(fir::fnacc::LaunchOp launchOp) {
  Region &region = launchOp.getRegion();
  if (region.empty())
    return fail(launchOp, "launch region is empty");

  Block &launchBlock = region.front();

  fir::DoLoopOp loop;
  for (Operation &op : launchBlock) {
    if (auto dl = dyn_cast<fir::DoLoopOp>(op)) {
      if (loop)
        return fail(&op, "1-D recognition expected exactly one top-level loop");
      loop = dl;
    }
  }

  if (!loop)
    return fail(launchOp, "1-D recognition found no top-level loop");

  std::string loopReason;
  if (!verifyLoopLowerBoundAndStep(loop, "1-D", loopReason))
    return fail(loop.getOperation(), loopReason);

  ElementwiseExtentSource extentX = getLoopExtentSource(loop);

  if (extentX.kind == ElementwiseExtentSourceKind::Unknown)
    return fail(loop.getOperation(), "could not determine 1-D loop extent");

  Value indMemref = findInductionMemref(loop);
  if (!indMemref)
    return fail(loop.getOperation(),
                "could not find induction variable storage in 1-D loop");

  ElementwiseKernel k;
  k.rank = 1;
  k.loop1D = loop;
  k.extentX = extentX;
  k.innerIndMemref = indMemref;

  markLoopBounds(k, launchOp, loop);
  markInductionStore(k, loop, indMemref);
  markPostLoopInductionUpdate(k, launchOp, loop, indMemref);

  std::string reason;
  if (!collectArrayAccesses1D(k, loop, indMemref, reason))
    return fail(loop.getOperation(), reason);

  return ElementwiseRecognitionResult::success(std::move(k));
}

static ElementwiseRecognitionResult recognize2D(fir::fnacc::LaunchOp launchOp) {
  Region &region = launchOp.getRegion();
  if (region.empty())
    return fail(launchOp, "launch region is empty");

  Block &launchBlock = region.front();

  fir::DoLoopOp outer;
  for (Operation &op : launchBlock) {
    if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
      if (outer)
        return fail(&op, "2-D recognition expected exactly one outer loop");
      outer = loop;
    }
  }

  if (!outer)
    return fail(launchOp, "2-D recognition found no outer loop");

  Block *outerBody = outer.getBody();
  if (!outerBody)
    return fail(outer.getOperation(), "outer loop has no body");

  fir::DoLoopOp inner;
  for (Operation &op : outerBody->getOperations()) {
    if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
      if (inner)
        return fail(&op, "2-D recognition expected exactly one inner loop");
      inner = loop;
    }
  }

  if (!inner)
    return fail(outer.getOperation(), "2-D recognition found no inner loop");

  std::string loopReason;
  if (!verifyLoopLowerBoundAndStep(outer, "outer", loopReason))
    return fail(outer.getOperation(), loopReason);

  if (!verifyLoopLowerBoundAndStep(inner, "inner", loopReason))
    return fail(inner.getOperation(), loopReason);

  ElementwiseExtentSource extentY = getLoopExtentSource(outer);
  ElementwiseExtentSource extentX = getLoopExtentSource(inner);

  if (extentY.kind == ElementwiseExtentSourceKind::Unknown)
    return fail(outer.getOperation(), "could not determine outer loop extent");

  if (extentX.kind == ElementwiseExtentSourceKind::Unknown)
    return fail(inner.getOperation(), "could not determine inner loop extent");

  Value outerIndMemref = findInductionMemref(outer);
  Value innerIndMemref = findInductionMemref(inner);

  if (!outerIndMemref)
    return fail(outer.getOperation(),
                "could not find outer induction variable storage");

  if (!innerIndMemref)
    return fail(inner.getOperation(),
                "could not find inner induction variable storage");

  ElementwiseKernel k;
  k.rank = 2;
  k.outerLoop = outer;
  k.innerLoop = inner;
  k.extentY = extentY;
  k.extentX = extentX;
  k.outerIndMemref = outerIndMemref;
  k.innerIndMemref = innerIndMemref;

  markLoopBounds(k, launchOp, outer);
  markLoopBounds(k, launchOp, inner);
  markInductionStore(k, outer, outerIndMemref);
  markInductionStore(k, inner, innerIndMemref);
  markPostLoopInductionUpdate(k, launchOp, outer, outerIndMemref);

  std::string reason;
  if (!collectArrayAccesses2D(k, inner, innerIndMemref, outerIndMemref, reason))
    return fail(inner.getOperation(), reason);

  return ElementwiseRecognitionResult::success(std::move(k));
}

} // namespace

bool isSupportedElementwiseCompute(Operation *op) {
  return op &&
         isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp>(op);
}

ElementwiseRecognitionResult
recognizeElementwiseKernel(fir::fnacc::LaunchOp launchOp) {
  // Reductions are explicitly marked by lowering with fnacc.reduction_slots.
  // Do not try reduction recognition on ordinary elementwise/matmul launches,
  // otherwise diagnostics become noisy and misleading.
  if (launchOp->hasAttr("fnacc.reduction_slots")) {
    auto rRed = recognizeReduction1D(launchOp);
    if (rRed.succeeded())
      return validateRecognizedKernel(launchOp, std::move(rRed));

    std::string reason = "not a supported FNACC reduction kernel; ";
    reason += rRed.getFailure().reason;
    return fail(launchOp, reason);
  }

  // Try matmul first because it is also a 2-D loop nest.
  auto rMatmul = recognizeMatMul2D(launchOp);
  if (rMatmul.succeeded())
    return validateRecognizedKernel(launchOp, std::move(rMatmul));

  // Try 2-D before 1-D because 2-D kernels have one top-level loop too.
  auto r2 = recognize2D(launchOp);
  if (r2.succeeded())
    return validateRecognizedKernel(launchOp, std::move(r2));

  auto r1 = recognize1D(launchOp);
  if (r1.succeeded())
    return validateRecognizedKernel(launchOp, std::move(r1));

  std::string reason = "not a supported FNACC kernel; ";
  reason += "matmul failure: ";
  reason += rMatmul.getFailure().reason.empty()
                ? "<unknown matmul recognition failure>"
                : rMatmul.getFailure().reason;
  reason += "; 2-D failure: ";
  reason += r2.getFailure().reason;
  reason += "; 1-D failure: ";
  reason += r1.getFailure().reason;

  return fail(launchOp, reason);
}

namespace {

static constexpr llvm::StringLiteral kKernelIdAttrName = "fnacc.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fnacc.kernel_name";

static int32_t getPlannedKernelId(fir::fnacc::LaunchOp launchOp,
                                  int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName))
    return static_cast<int32_t>(attr.getInt());
  return fallbackId;
}

static std::string getPlannedKernelName(fir::fnacc::LaunchOp launchOp,
                                        int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<StringAttr>(kKernelNameAttrName))
    return attr.getValue().str();
  return "fnacc_kernel_" + std::to_string(fallbackId);
}

static int32_t getPlannedParallelSubgroups(const ElementwiseKernel &kernel,
                                           int32_t requestedParallelSubgroups) {
  bool isF64ScalarElementwise =
      kernel.elementType == ElementType::F64 && !kernel.scalarRefs.empty() &&
      (kernel.kind == ElementwiseKernelKind::Saxpy1D ||
       kernel.kind == ElementwiseKernelKind::Expr1D ||
       kernel.kind == ElementwiseKernelKind::Expr2D);

  if (isF64ScalarElementwise || isReductionKernelKind(kernel.kind) ||
      kernel.kind == ElementwiseKernelKind::MatMul2D)
    return requestedParallelSubgroups;

  return 1;
}

static llvm::SmallVector<unsigned>
getKernelParameterSlotsForValue(const ElementwiseKernel &kernel, Value value) {
  llvm::SmallVector<unsigned> slots;

  for (unsigned i = 0; i < kernel.readArrays.size(); ++i)
    if (kernel.readArrays[i] == value)
      slots.push_back(i);

  if (kernel.writeArray == value)
    slots.push_back(kernel.readArrays.size());

  unsigned scalarBaseSlot = kernel.readArrays.size() + 1;
  for (unsigned i = 0; i < kernel.scalarRefs.size(); ++i)
    if (kernel.scalarRefs[i] == value)
      slots.push_back(scalarBaseSlot + i);

  return slots;
}

static bool isReductionMetadataPackSlot(fir::fnacc::LaunchOp launchOp,
                                        unsigned packIndex) {
  auto slotsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_slots");
  if (!slotsAttr)
    return false;

  for (int32_t slot : slotsAttr.asArrayRef())
    if (slot >= 0 && static_cast<unsigned>(slot) == packIndex)
      return true;

  return false;
}

static void appendABIParameter(FNACCKernelABI &abi,
                               FNACCKernelParameterRole role,
                               FNACCKernelParameterPassing passing,
                               ElementType elementType, llvm::StringRef name) {
  FNACCKernelParameter parameter;
  parameter.slot = abi.parameters.size();
  parameter.role = role;
  parameter.passing = passing;
  parameter.elementType = elementType;
  parameter.name = name.str();
  abi.parameters.push_back(std::move(parameter));
}

static FNACCKernelABI buildKernelABI(fir::fnacc::LaunchOp launchOp,
                                     const ElementwiseKernel &kernel) {
  FNACCKernelABI abi;
  bool isReduction = isReductionKernelKind(kernel.kind);

  for (unsigned i = 0; i < kernel.readArrays.size(); ++i)
    appendABIParameter(abi, FNACCKernelParameterRole::Read,
                       FNACCKernelParameterPassing::DevicePointer,
                       kernel.elementType, "read" + std::to_string(i));

  if (isReduction) {
    appendABIParameter(abi, FNACCKernelParameterRole::Partials,
                       FNACCKernelParameterPassing::DevicePointer,
                       kernel.elementType, "partials");
    appendABIParameter(abi, FNACCKernelParameterRole::ExtentX,
                       FNACCKernelParameterPassing::Value, ElementType::I32,
                       "extent_x");
  } else {
    appendABIParameter(abi, FNACCKernelParameterRole::Write,
                       FNACCKernelParameterPassing::DevicePointer,
                       kernel.elementType, "write");

    for (unsigned i = 0; i < kernel.scalarRefs.size(); ++i)
      appendABIParameter(abi, FNACCKernelParameterRole::Scalar,
                         FNACCKernelParameterPassing::Value, kernel.elementType,
                         "scalar" + std::to_string(i));

    appendABIParameter(abi, FNACCKernelParameterRole::ExtentX,
                       FNACCKernelParameterPassing::Value, ElementType::I32,
                       "extent_x");

    if (kernel.rank == 2) {
      appendABIParameter(abi, FNACCKernelParameterRole::ExtentY,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "extent_y");

      if (kernel.kind == ElementwiseKernelKind::MatMul2D)
        appendABIParameter(abi, FNACCKernelParameterRole::ExtentZ,
                           FNACCKernelParameterPassing::Value, ElementType::I32,
                           "extent_k");
    }
  }

  auto packVars = launchOp.getPackVars();
  llvm::ArrayRef<int32_t> targets = launchOp.getPackTargets();

  for (auto [packIndex, packValue] : llvm::enumerate(packVars)) {
    if (isReductionMetadataPackSlot(launchOp, packIndex))
      continue;

    llvm::SmallVector<unsigned> slots =
        getKernelParameterSlotsForValue(kernel, packValue);
    if (slots.empty()) {
      launchOp.emitWarning() << "PACK variable #" << packIndex
                             << " was not used by recognized FNACC kernel body";
      continue;
    }

    if (packIndex >= targets.size()) {
      launchOp.emitWarning("PACK target list shorter than PACK var list");
      continue;
    }

    for (unsigned slot : slots)
      abi.packBindings.push_back({slot, targets[packIndex]});
  }

  return abi;
}

static FNACCKernelABI buildReductionStageABI(ElementType elementType) {
  FNACCKernelABI abi;
  appendABIParameter(abi, FNACCKernelParameterRole::Read,
                     FNACCKernelParameterPassing::DevicePointer, elementType,
                     "input");
  appendABIParameter(abi, FNACCKernelParameterRole::Partials,
                     FNACCKernelParameterPassing::DevicePointer, elementType,
                     "output");
  appendABIParameter(abi, FNACCKernelParameterRole::ExtentX,
                     FNACCKernelParameterPassing::Value, ElementType::I32,
                     "extent_x");
  return abi;
}

static FNACCTileShape getPlannedTileShape(fir::fnacc::LaunchOp launchOp,
                                          const ElementwiseKernel &kernel) {
  llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();
  FNACCTileShape tile;

  if (kernel.rank == 2) {
    tile.x = tiles.size() >= 1 ? tiles[0] : 16;
    tile.y = tiles.size() >= 2 ? tiles[1] : 16;
    if (kernel.kind == ElementwiseKernelKind::MatMul2D)
      tile.z = tiles.size() >= 3
                   ? tiles[2]
                   : (kernel.elementType == ElementType::F64 ? 8 : 32);
  } else {
    tile.x = tiles.empty() ? 1024 : tiles[0];
  }

  return tile;
}

} // namespace

FNACCKernelPlanResult FNACCKernelPlanResult::success(FNACCKernelPlan plan) {
  FNACCKernelPlanResult result;
  result.plan.emplace(std::move(plan));
  return result;
}

FNACCKernelPlanResult FNACCKernelPlanResult::failure(Operation *where,
                                                     std::string reason) {
  FNACCKernelPlanResult result;
  result.failureInfo.where = where;
  result.failureInfo.reason = std::move(reason);
  return result;
}

FNACCKernelPlan FNACCKernelPlanResult::takePlan() {
  assert(plan && "cannot take a failed FNACC kernel plan");
  FNACCKernelPlan result = std::move(*plan);
  plan.reset();
  return result;
}

bool isReductionKernelKind(ElementwiseKernelKind kind) {
  return kind == ElementwiseKernelKind::ReductionSum1D ||
         kind == ElementwiseKernelKind::ReductionDot1D ||
         kind == ElementwiseKernelKind::ReductionProduct1D ||
         kind == ElementwiseKernelKind::ReductionMin1D ||
         kind == ElementwiseKernelKind::ReductionMax1D;
}

llvm::StringRef fnaccKernelKindName(ElementwiseKernelKind kind) {
  switch (kind) {
  case ElementwiseKernelKind::BinaryArrayArray:
    return "binary";
  case ElementwiseKernelKind::Saxpy1D:
    return "saxpy1d";
  case ElementwiseKernelKind::Expr1D:
    return "expr1d";
  case ElementwiseKernelKind::Expr2D:
    return "expr2d";
  case ElementwiseKernelKind::MatMul2D:
    return "matmul2d";
  case ElementwiseKernelKind::ReductionSum1D:
    return "reduction_sum1d";
  case ElementwiseKernelKind::ReductionDot1D:
    return "reduction_dot1d";
  case ElementwiseKernelKind::ReductionProduct1D:
    return "reduction_product1d";
  case ElementwiseKernelKind::ReductionMin1D:
    return "reduction_min1d";
  case ElementwiseKernelKind::ReductionMax1D:
    return "reduction_max1d";
  }
  llvm_unreachable("unknown FNACC kernel kind");
}

FNACCKernelPlanResult
buildFNACCKernelPlan(fir::fnacc::LaunchOp launchOp, int32_t fallbackId,
                     int32_t nextSyntheticKernelId,
                     const FNACCKernelPlanOptions &options) {
  ElementwiseRecognitionResult recognition =
      recognizeElementwiseKernel(launchOp);
  if (recognition.failed())
    return FNACCKernelPlanResult::failure(recognition.getFailure().where,
                                          recognition.getFailure().reason);

  ElementwiseKernel kernel = std::move(recognition.getKernel());

  FNACCKernelPlan plan;
  plan.launchOp = launchOp;
  plan.id = getPlannedKernelId(launchOp, fallbackId);
  plan.name = getPlannedKernelName(launchOp, plan.id);
  plan.kernel = std::move(kernel);
  plan.schedule.tile = getPlannedTileShape(launchOp, plan.kernel);
  plan.schedule.parallelSubgroups = getPlannedParallelSubgroups(
      plan.kernel, options.requestedParallelSubgroups);
  plan.schedule.subgroupWidth = options.subgroupWidth;
  plan.schedule.pipelineStages = options.pipelineStages;
  plan.schedule.f64MatmulStrategy = options.f64MatmulStrategy;
  plan.abi = buildKernelABI(launchOp, plan.kernel);

  if (isReductionKernelKind(plan.kernel.kind)) {
    FNACCReductionStagePlan stage;
    stage.id = nextSyntheticKernelId;
    stage.name = plan.name + "_reduce_stage";
    stage.reductionOperator = plan.kernel.reductionOperator;
    stage.elementType = plan.kernel.elementType;
    stage.abi = buildReductionStageABI(plan.kernel.elementType);
    plan.reductionStage = std::move(stage);
  }

  return FNACCKernelPlanResult::success(std::move(plan));
}

FNACCBackendSelection selectFNACCBackend(
    const FNACCKernelPlan &plan,
    llvm::ArrayRef<const FNACCCodegenBackend *> availableBackends,
    llvm::StringRef preferredBackend, llvm::StringRef fallbackBackend,
    bool allowFallback) {
  auto findBackend = [&](llvm::StringRef name) -> const FNACCCodegenBackend * {
    for (const FNACCCodegenBackend *backend : availableBackends)
      if (backend && backend->getName() == name)
        return backend;
    return nullptr;
  };

  if (preferredBackend.empty() || preferredBackend == "auto") {
    std::string unsupportedReasons;
    for (const FNACCCodegenBackend *backend : availableBackends) {
      if (!backend)
        continue;
      FNACCBackendSupport support = backend->querySupport(plan);
      if (support.supported)
        return {backend, false, {}};
      if (!unsupportedReasons.empty())
        unsupportedReasons += "; ";
      unsupportedReasons += backend->getName().str() + ": " + support.reason;
    }

    return {nullptr, false,
            unsupportedReasons.empty()
                ? "no FNACC code-generation backends are registered"
                : "no registered FNACC backend supports this kernel: " +
                      unsupportedReasons};
  }

  const FNACCCodegenBackend *preferred = findBackend(preferredBackend);
  std::string preferredFailure;
  if (!preferred) {
    preferredFailure =
        "requested backend '" + preferredBackend.str() + "' is not registered";
  } else {
    FNACCBackendSupport support = preferred->querySupport(plan);
    if (support.supported)
      return {preferred, false, {}};
    preferredFailure = "backend '" + preferredBackend.str() +
                       "' does not support this kernel: " + support.reason;
  }

  if (!allowFallback)
    return {nullptr, false, std::move(preferredFailure)};

  const FNACCCodegenBackend *fallback = findBackend(fallbackBackend);
  if (!fallback)
    return {nullptr, false,
            preferredFailure + "; fallback backend '" + fallbackBackend.str() +
                "' is not registered"};

  FNACCBackendSupport fallbackSupport = fallback->querySupport(plan);
  if (!fallbackSupport.supported)
    return {nullptr, false,
            preferredFailure + "; fallback backend '" + fallbackBackend.str() +
                "' does not support this kernel: " + fallbackSupport.reason};

  return {fallback, true,
          preferredFailure + "; falling back to '" + fallbackBackend.str() +
              "'"};
}

} // namespace fir::fnacc

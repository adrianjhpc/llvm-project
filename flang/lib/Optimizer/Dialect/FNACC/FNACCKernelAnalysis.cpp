#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"

#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

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

static void markConsumed(ElementwiseKernel &kernel, Operation *op) {
  if (!op)
    return;

  if (!llvm::is_contained(kernel.consumedOps, op))
    kernel.consumedOps.push_back(op);
}

static void markConsumedValueDefinition(ElementwiseKernel &kernel,
                                        Value value) {
  if (Operation *definingOp = value.getDefiningOp())
    markConsumed(kernel, definingOp);
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

static void markBackwardSlice(ElementwiseKernel &kernel, Operation *root,
                              fir::fnacc::LaunchOp launchOp) {
  if (!root || !launchOp->isProperAncestor(root) ||
      llvm::is_contained(kernel.consumedOps, root))
    return;

  markConsumed(kernel, root);

  for (Value operand : root->getOperands()) {
    if (Operation *def = operand.getDefiningOp())
      markBackwardSlice(kernel, def, launchOp);
  }
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

static std::optional<Value> getScalarRealRefFromValue(Value v) {
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

  if (!eleTy.isF32() && !eleTy.isF64())
    return std::nullopt;

  return memref;
}

static fir::fnacc::ElementType getScalarRealRefElementType(Value memref) {
  auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
  if (!refTy)
    return fir::fnacc::ElementType::Unknown;

  Type eleTy = refTy.getEleTy();

  if (eleTy.isF32())
    return fir::fnacc::ElementType::F32;

  if (eleTy.isF64())
    return fir::fnacc::ElementType::F64;

  return fir::fnacc::ElementType::Unknown;
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

static fir::fnacc::ElementType getRealArrayElementType(Value v) {
  Type type = unwrapArrayStorageType(v.getType());
  auto arrTy = dyn_cast<fir::SequenceType>(type);

  if (!arrTy)
    return fir::fnacc::ElementType::Unknown;

  Type eleTy = arrTy.getEleTy();

  if (eleTy.isF32())
    return fir::fnacc::ElementType::F32;

  if (eleTy.isF64())
    return fir::fnacc::ElementType::F64;

  return fir::fnacc::ElementType::Unknown;
}

static bool inferAndCheckElementType(ElementwiseKernel &k,
                                     std::string &reason) {
  if (!k.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  fir::fnacc::ElementType type = getRealArrayElementType(k.writeArray);
  if (type == fir::fnacc::ElementType::Unknown) {
    reason = "write array must be real(4) or real(8)";
    return false;
  }

  for (Value read : k.readArrays) {
    fir::fnacc::ElementType readType = getRealArrayElementType(read);
    if (readType != type) {
      reason = "all read/write arrays must have the same real element type";
      return false;
    }
  }

  for (Value scalar : k.scalarRefs) {
    if (getScalarRealRefElementType(scalar) != type) {
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

static std::optional<Value>
getScalarRealRefFromValue(Value v, fir::fnacc::ElementType expectedType) {
  v = stripFirConvert(v);

  auto load = v.getDefiningOp<fir::LoadOp>();
  if (!load)
    return std::nullopt;

  Value memref = load.getMemref();

  if (memref.getDefiningOp<fir::ArrayCoorOp>())
    return std::nullopt;

  auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
  if (!refTy)
    return std::nullopt;

  Type eleTy = refTy.getEleTy();

  if (expectedType == fir::fnacc::ElementType::Unknown) {
    if (eleTy.isF32() || eleTy.isF64())
      return memref;
    return std::nullopt;
  }

  if (expectedType == fir::fnacc::ElementType::F32 && eleTy.isF32())
    return memref;

  if (expectedType == fir::fnacc::ElementType::F64 && eleTy.isF64())
    return memref;

  return std::nullopt;
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

static bool isRealElementType(Type type, ElementType expected) {
  if (expected == ElementType::F32)
    return type.isF32();
  if (expected == ElementType::F64)
    return type.isF64();
  return type.isF32() || type.isF64();
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

static bool isSupportedExpressionFloatType(Type type) {
  return type.isF32() || type.isF64();
}

static std::unique_ptr<ElementwiseExpr>
recognizeElementwiseExpr(Value value, const ArrayAccessInfo &accesses,
                         ElementwiseKernel &kernel, std::string &reason) {
  value = stripFirConvert(value);

  // Array element load previously collected from a recognized array_coor.
  int readIndex = findReadValueIndex(accesses.readValues, value);
  if (readIndex >= 0) {
    Value arrayBase = accesses.readArrays[readIndex];
    getOrAddValueIndex(kernel.readArrays, arrayBase);

    auto expression = makeExpr(ElementwiseExprKind::ArrayLoad);
    expression->source = arrayBase;
    return expression;
  }

  // Fortran scalar real variable captured by reference.
  if (auto scalarRef = getScalarRealRefFromValue(value)) {
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

  Operation *operation = value.getDefiningOp();
  if (!operation) {
    reason = "elementwise expression value has no defining operation";
    return nullptr;
  }

  Type resultType = value.getType();

  // The only supported non-floating expression result is an i1 predicate.
  if (!isSupportedExpressionFloatType(resultType) && !resultType.isInteger(1)) {
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

  // Use operation names for min/max because exact generated C++ class names
  // differ across MLIR revisions.
  StringRef operationName = operation->getName().getStringRef();

  if (operationName == "arith.minimumf" || operationName == "arith.minnumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point minimum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MinF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
  }

  if (operationName == "arith.maximumf" || operationName == "arith.maxnumf") {
    if (operation->getNumOperands() != 2) {
      reason = "floating-point maximum operation is not binary";
      return nullptr;
    }

    return recognizeBinaryElementwiseExpr(
        ElementwiseExprKind::MaxF, operation->getOperand(0),
        operation->getOperand(1), accesses, kernel, reason);
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
      reason = "elementwise select true value is not a real value";
      return nullptr;
    }

    auto falseValue = recognizeElementwiseExpr(select.getFalseValue(), accesses,
                                               kernel, reason);
    if (!falseValue)
      return nullptr;

    if (falseValue->resultKind != ElementwiseExprResultKind::Element) {
      reason = "elementwise select false value is not a real value";
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
    reason = "expression tree supports at most three scalar f32 values";
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

  if (info.readArrays.size() < minReads || info.readArrays.size() > maxReads ||
      info.readValues.size() < minReads || info.readValues.size() > maxReads) {
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
    reason = "stored value is not a supported binary floating-point op";
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
      scalarRef = getScalarRealRefFromValue(mulRhs);
    } else {
      scaledArrayIndex = findReadValueIndex(info.readValues, mulRhs);
      if (scaledArrayIndex >= 0)
        scalarRef = getScalarRealRefFromValue(mulLhs);
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

  if (info.readArrays.empty() || info.readArrays.size() > 3) {
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

  fir::fnacc::ElementType type = getRealArrayElementType(k.readArrays[0]);

  if (type == fir::fnacc::ElementType::Unknown) {
    reason = "reduction read array must be real(4) or real(8)";
    return false;
  }

  for (Value read : k.readArrays) {
    if (getRealArrayElementType(read) != type) {
      reason = "all reduction read arrays must have same element type";
      return false;
    }
  }

  if (!k.reductionScalarRef) {
    reason = "reduction has no scalar result reference";
    return false;
  }

  if (getScalarRealRefElementType(k.reductionScalarRef) != type) {
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
    return name == "arith.addf";
  case ReductionOperator::Multiply:
    return name == "arith.mulf";
  case ReductionOperator::Min:
    if (name == "arith.minimumf" || name == "arith.minnumf")
      return true;
    break;
  case ReductionOperator::Max:
    if (name == "arith.maximumf" || name == "arith.maxnumf")
      return true;
    break;
  }

  auto select = dyn_cast<arith::SelectOp>(op);
  if (!select || (reductionOp != ReductionOperator::Min &&
                  reductionOp != ReductionOperator::Max))
    return false;

  auto compare = select.getCondition().getDefiningOp<arith::CmpFOp>();
  if (!compare)
    return false;

  Value cmpLhs = stripFirConvert(compare.getLhs());
  Value cmpRhs = stripFirConvert(compare.getRhs());
  Value trueValue = stripFirConvert(select.getTrueValue());
  Value falseValue = stripFirConvert(select.getFalseValue());

  bool direct = trueValue == cmpLhs && falseValue == cmpRhs;
  bool reversed = trueValue == cmpRhs && falseValue == cmpLhs;
  if (!direct && !reversed)
    return false;

  using Predicate = arith::CmpFPredicate;
  Predicate predicate = compare.getPredicate();
  bool less = predicate == Predicate::OLT || predicate == Predicate::OLE ||
              predicate == Predicate::ULT || predicate == Predicate::ULE;
  bool greater = predicate == Predicate::OGT || predicate == Predicate::OGE ||
                 predicate == Predicate::UGT || predicate == Predicate::UGE;

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

  auto add = value.getDefiningOp<arith::AddFOp>();
  if (!add) {
    reason = "dot reduction store value is not arith.addf";
    return false;
  }

  Value lhs = stripFirConvert(add.getLhs());
  Value rhs = stripFirConvert(add.getRhs());

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

  auto mul = maybeMul.getDefiningOp<arith::MulFOp>();
  if (!mul) {
    reason = "dot reduction term is not arith.mulf";
    return false;
  }

  Value mulLhs = stripFirConvert(mul.getLhs());
  Value mulRhs = stripFirConvert(mul.getRhs());

  int lhsIndex = findReductionLoadedValueIndex(loads, mulLhs);
  int rhsIndex = findReductionLoadedValueIndex(loads, mulRhs);

  if (lhsIndex < 0 || rhsIndex < 0 || lhsIndex == rhsIndex) {
    reason = "dot reduction multiply operands are not two distinct array loads";
    return false;
  }

  arrayA = loads[lhsIndex].arrayBase;
  arrayB = loads[rhsIndex].arrayBase;
  computeOp = mul.getOperation();
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
         isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(op);
}

ElementwiseRecognitionResult
recognizeElementwiseKernel(fir::fnacc::LaunchOp launchOp) {
  // Reductions are explicitly marked by lowering with fnacc.reduction_slots.
  // Do not try reduction recognition on ordinary elementwise/matmul launches,
  // otherwise diagnostics become noisy and misleading.
  if (launchOp->hasAttr("fnacc.reduction_slots")) {
    auto rRed = recognizeReduction1D(launchOp);
    if (rRed.succeeded())
      return rRed;

    std::string reason = "not a supported FNACC reduction kernel; ";
    reason += rRed.getFailure().reason;
    return fail(launchOp, reason);
  }

  // Try matmul first because it is also a 2-D loop nest.
  auto rMatmul = recognizeMatMul2D(launchOp);
  if (rMatmul.succeeded())
    return rMatmul;

  // Try 2-D before 1-D because 2-D kernels have one top-level loop too.
  auto r2 = recognize2D(launchOp);
  if (r2.succeeded())
    return r2;

  auto r1 = recognize1D(launchOp);
  if (r1.succeeded())
    return r1;

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

} // namespace fir::fnacc

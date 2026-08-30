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
#include <functional>
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

static fir::fnacc::ElementwiseExtentSource getIntegerSource(Value value) {
  value = stripFirConvert(value);

  llvm::APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)) &&
      constant.getBitWidth() <= 64) {
    ElementwiseExtentSource source;
    source.kind = ElementwiseExtentSourceKind::ConstantInteger;
    source.constantValue = constant.getSExtValue();
    return source;
  }

  if (auto ref = getIntegerRefFromLoadLike(value)) {
    ElementwiseExtentSource source;
    source.kind = ElementwiseExtentSourceKind::LoadIntegerRef;
    source.value = *ref;
    return source;
  }

  ElementwiseExtentSource source;
  source.kind = ElementwiseExtentSourceKind::Value;
  source.value = value;
  return source;
}

static ElementwiseExtentSource getLoopLowerSource(fir::DoLoopOp loop) {
  return getIntegerSource(loop.getLowerBound());
}

static bool isAcceptedStructuralOperation(Operation *op) {
  return isa<fir::ResultOp, fir::fnacc::TerminatorOp, arith::ConstantOp,
             arith::ConstantIndexOp, fir::ShapeOp, fir::ShapeShiftOp>(op);
}

static bool sameValueAfterFirConvert(Value lhs, Value rhs) {
  return lhs && rhs && stripFirConvert(lhs) == stripFirConvert(rhs);
}

static bool scalarReferenceIsUsedAfterLaunch(Value reference,
                                             fir::fnacc::LaunchOp launchOp);

static Value getScalarStorageRoot(Value value) {
  while (true) {
    value = stripFirConvert(value);
    auto declareOp = value.getDefiningOp<fir::DeclareOp>();
    if (!declareOp)
      return value;
    value = declareOp.getMemref();
  }
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
  if (!arrayCoor) {
    // An iteration-local temporary which is assigned but never read has no
    // observable value after the parallel loop. CloverLeaf contains a few of
    // these source-level diagnostics temporaries (for example
    // min_cell_volume). Do not turn such dead stores into shared kernel
    // state, but permit host lowering to erase them.
    auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
    Type elementType = refTy ? refTy.getEleTy() : Type{};
    bool supportedScalar =
        elementType && (elementType.isF32() || elementType.isF64() ||
                        elementType.isInteger(8) || elementType.isInteger(16) ||
                        elementType.isInteger(32) || elementType.isInteger(64));
    if (supportedScalar) {
      // Only an alloca-backed compiler temporary may be discarded as a dead
      // iteration-private value. A dummy argument, global, or other
      // host-visible scalar is an observable kernel output even when this
      // launch does not load it. In particular, do not accept an arbitrary
      // side-effecting scalar store merely because the reference has no load
      // users.
      Value storageRoot = getScalarStorageRoot(memref);
      if (!storageRoot.getDefiningOp<fir::AllocaOp>())
        return false;

      bool hasLoad = llvm::any_of(memref.getUsers(), [](Operation *user) {
        return isa<fir::LoadOp>(user);
      });
      auto launchOp = store->getParentOfType<fir::fnacc::LaunchOp>();
      return launchOp && !hasLoad &&
             !scalarReferenceIsUsedAfterLaunch(memref, launchOp) &&
             !scalarReferenceIsUsedAfterLaunch(storageRoot, launchOp);
    }
    return false;
  }

  if (sameArrayBase(arrayCoor.getMemref(), kernel.writeArray))
    return true;

  return llvm::any_of(kernel.writeArrays, [&](Value array) {
    return sameArrayBase(arrayCoor.getMemref(), array);
  });
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

static void populateVariadicArrayArguments(ElementwiseKernel &kernel) {
  if (!usesVariadicLaunchABI(kernel.kind) || !kernel.outputs.empty() ||
      !kernel.reductionOutputs.empty())
    return;

  kernel.arrayArguments.clear();
  auto addArray = [&](Value array, bool read, bool write) {
    for (ElementwiseArrayArgument &argument : kernel.arrayArguments) {
      if (!sameArrayBase(argument.array, array))
        continue;
      argument.read |= read;
      argument.write |= write;
      if (argument.elementType == ElementType::Unknown)
        argument.elementType = kernel.elementType;
      return;
    }
    kernel.arrayArguments.push_back({array, read, write, kernel.elementType});
  };

  for (Value array : kernel.readArrays)
    addArray(array, true, false);
  if (kernel.writeArray)
    addArray(kernel.writeArray, false, true);
}

static ElementwiseRecognitionResult
validateRecognizedKernel(fir::fnacc::LaunchOp launchOp,
                         ElementwiseRecognitionResult result) {
  ElementwiseKernel kernel = std::move(result.getKernel());
  populateVariadicArrayArguments(kernel);

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

static bool verifyLoopLowerBoundAndStep(fir::DoLoopOp loop, StringRef loopName,
                                        std::string &reason) {
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

static std::optional<int64_t> getIndexConstant(Value value) {
  value = stripFirConvert(value);
  llvm::APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)) ||
      constant.getBitWidth() > 64)
    return std::nullopt;
  return constant.getSExtValue();
}

static std::optional<Value> getReadOnlyScalarCaptureRef(Value value);

struct AffineIndexMatch {
  int64_t coefficient = 0;
  Value baseRef;
  int64_t baseCoefficient = 0;
  int64_t offset = 0;
};

static std::optional<AffineIndexMatch>
combineAffineIndexMatches(const AffineIndexMatch &lhs,
                          const AffineIndexMatch &rhs, int64_t rhsSign) {
  if (lhs.baseRef && rhs.baseRef)
    return std::nullopt;

  AffineIndexMatch result;
  result.coefficient = lhs.coefficient + rhsSign * rhs.coefficient;
  result.baseRef = lhs.baseRef ? lhs.baseRef : rhs.baseRef;
  result.baseCoefficient =
      lhs.baseRef ? lhs.baseCoefficient : rhsSign * rhs.baseCoefficient;
  result.offset = lhs.offset + rhsSign * rhs.offset;
  return result;
}

static std::optional<AffineIndexMatch>
matchAffineIndexExpression(Value value, Value inductionMemref) {
  value = stripFirConvert(value);

  if (auto constant = getIndexConstant(value)) {
    AffineIndexMatch result;
    result.offset = *constant;
    return result;
  }

  if (auto load = value.getDefiningOp<fir::LoadOp>()) {
    if (load.getMemref() == inductionMemref) {
      AffineIndexMatch result;
      result.coefficient = 1;
      return result;
    }

    std::optional<Value> capture = getReadOnlyScalarCaptureRef(value);
    if (!capture)
      return std::nullopt;
    auto refTy = dyn_cast<fir::ReferenceType>(capture->getType());
    if (!refTy)
      return std::nullopt;
    Type elementType = refTy.getEleTy();
    if (!elementType.isInteger(8) && !elementType.isInteger(16) &&
        !elementType.isInteger(32) && !elementType.isInteger(64))
      return std::nullopt;

    AffineIndexMatch result;
    result.baseRef = *capture;
    result.baseCoefficient = 1;
    return result;
  }

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = matchAffineIndexExpression(add.getLhs(), inductionMemref);
    auto rhs = matchAffineIndexExpression(add.getRhs(), inductionMemref);
    if (lhs && rhs)
      return combineAffineIndexMatches(*lhs, *rhs, 1);
  }

  if (auto subtract = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = matchAffineIndexExpression(subtract.getLhs(), inductionMemref);
    auto rhs = matchAffineIndexExpression(subtract.getRhs(), inductionMemref);
    if (lhs && rhs)
      return combineAffineIndexMatches(*lhs, *rhs, -1);
  }

  return std::nullopt;
}

/// Match `coefficient * induction + base + constant` after FIR converts.
/// Coefficients are limited to +1 and -1, and at most one host-visible integer
/// base is accepted. This covers ordinary stencil offsets as well as reversed
/// halo coordinates such as `x_min-j` and `left_xmax+1-j`.
static std::optional<AffineIndexMatch> matchAffineIndex(Value value,
                                                        Value inductionMemref) {
  std::optional<AffineIndexMatch> result =
      matchAffineIndexExpression(value, inductionMemref);
  if (!result || (result->coefficient != 1 && result->coefficient != -1) ||
      (result->baseRef && result->baseCoefficient != 1))
    return std::nullopt;
  return result;
}

static Value stripIndexExpressionWrappers(Value value) {
  while (true) {
    value = stripFirConvert(value);
    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getNumOperands() != 1 ||
        operation->getName().getStringRef() != "fir.no_reassoc")
      return value;
    value = operation->getOperand(0);
  }
}

static std::shared_ptr<ElementwiseIndexExpr>
makeIndexExpr(ElementwiseIndexExprKind kind) {
  auto expression = std::make_shared<ElementwiseIndexExpr>();
  expression->kind = kind;
  return expression;
}

static unsigned
getIndexExpressionLoopMask(const std::shared_ptr<ElementwiseIndexExpr> &expr) {
  if (!expr)
    return 0;
  if (expr->kind == ElementwiseIndexExprKind::LoopIndex)
    return 1u << expr->loopDimension;
  unsigned mask = 0;
  for (const auto &operand : expr->operands)
    mask |= getIndexExpressionLoopMask(operand);
  return mask;
}

static bool
sameIndexExpression(const std::shared_ptr<ElementwiseIndexExpr> &lhs,
                    const std::shared_ptr<ElementwiseIndexExpr> &rhs) {
  if (!lhs || !rhs)
    return lhs == rhs;
  if (lhs->kind != rhs->kind || lhs->loopDimension != rhs->loopDimension ||
      lhs->capture != rhs->capture || lhs->condition != rhs->condition ||
      lhs->constantValue != rhs->constantValue ||
      lhs->operands.size() != rhs->operands.size())
    return false;
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip_equal(lhs->operands, rhs->operands))
    if (!sameIndexExpression(lhsOperand, rhsOperand))
      return false;
  return true;
}

static bool
sameIndexExpressions(ArrayRef<std::shared_ptr<ElementwiseIndexExpr>> lhs,
                     ArrayRef<std::shared_ptr<ElementwiseIndexExpr>> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsExpression, rhsExpression] : llvm::zip_equal(lhs, rhs))
    if (!sameIndexExpression(lhsExpression, rhsExpression))
      return false;
  return true;
}

static bool regionStoresScalarReference(Region &region, Value reference);

static std::shared_ptr<ElementwiseIndexExpr>
matchGeneralIndexExpression(Value value, ArrayRef<Value> inductionMemrefs,
                            Operation *before,
                            llvm::SmallVectorImpl<Operation *> &consumedOps);

/// Reconstruct an iteration-private integer used as an array subscript.  Walk
/// backwards through straight-line stores and structured fir.if regions.  In
/// particular, this follows branch-local aliases such as `dif = donor`, where
/// both `dif` and `donor` are assigned inside the same arm of the conditional.
static std::shared_ptr<ElementwiseIndexExpr>
matchPrivateIndexValueBefore(Value reference, Operation *before,
                             ArrayRef<Value> inductionMemrefs,
                             llvm::SmallVectorImpl<Operation *> &consumedOps) {
  if (!before)
    return {};

  Block *block = before->getBlock();
  for (auto iterator = before->getIterator(); iterator != block->begin();) {
    --iterator;
    Operation *operation = &*iterator;

    if (auto store = dyn_cast<fir::StoreOp>(operation)) {
      if (!sameValueAfterFirConvert(store.getMemref(), reference))
        continue;
      if (!llvm::is_contained(consumedOps, store.getOperation()))
        consumedOps.push_back(store.getOperation());
      return matchGeneralIndexExpression(store.getValue(), inductionMemrefs,
                                         store.getOperation(), consumedOps);
    }

    auto ifOp = dyn_cast<fir::IfOp>(operation);
    if (!ifOp)
      continue;
    bool thenStores =
        regionStoresScalarReference(ifOp.getThenRegion(), reference);
    bool elseStores =
        regionStoresScalarReference(ifOp.getElseRegion(), reference);
    if (!thenStores && !elseStores)
      continue;

    if (!llvm::is_contained(consumedOps, ifOp.getOperation()))
      consumedOps.push_back(ifOp.getOperation());

    auto matchArm = [&](Region &region,
                        bool stores) -> std::shared_ptr<ElementwiseIndexExpr> {
      if (!stores)
        return matchPrivateIndexValueBefore(reference, ifOp.getOperation(),
                                            inductionMemrefs, consumedOps);
      if (region.empty())
        return {};
      return matchPrivateIndexValueBefore(reference,
                                          region.front().getTerminator(),
                                          inductionMemrefs, consumedOps);
    };

    auto trueValue = matchArm(ifOp.getThenRegion(), thenStores);
    auto falseValue = matchArm(ifOp.getElseRegion(), elseStores);
    if (!trueValue || !falseValue)
      return {};

    auto expression = makeIndexExpr(ElementwiseIndexExprKind::Select);
    expression->condition = ifOp.getCondition();
    expression->operands.push_back(std::move(trueValue));
    expression->operands.push_back(std::move(falseValue));
    return expression;
  }

  Operation *parent = block->getParentOp();
  if (!parent || isa<fir::DoLoopOp, fir::fnacc::LaunchOp>(parent))
    return {};
  return matchPrivateIndexValueBefore(reference, parent, inductionMemrefs,
                                      consumedOps);
}

static std::shared_ptr<ElementwiseIndexExpr>
matchGeneralIndexExpression(Value value, ArrayRef<Value> inductionMemrefs,
                            Operation *before,
                            llvm::SmallVectorImpl<Operation *> &consumedOps) {
  value = stripIndexExpressionWrappers(value);

  if (auto constant = getIndexConstant(value)) {
    auto expression = makeIndexExpr(ElementwiseIndexExprKind::Constant);
    expression->constantValue = *constant;
    return expression;
  }

  if (auto load = value.getDefiningOp<fir::LoadOp>()) {
    for (auto [dimension, memref] : llvm::enumerate(inductionMemrefs)) {
      if (!sameValueAfterFirConvert(load.getMemref(), memref))
        continue;
      auto expression = makeIndexExpr(ElementwiseIndexExprKind::LoopIndex);
      expression->loopDimension = dimension;
      return expression;
    }

    if (auto capture = getReadOnlyScalarCaptureRef(value)) {
      auto referenceType = dyn_cast<fir::ReferenceType>(capture->getType());
      Type type = referenceType ? referenceType.getEleTy() : Type{};
      if (type && (type.isInteger(8) || type.isInteger(16) ||
                   type.isInteger(32) || type.isInteger(64))) {
        auto expression = makeIndexExpr(ElementwiseIndexExprKind::Capture);
        expression->capture = *capture;
        return expression;
      }
    }

    if (auto expression = matchPrivateIndexValueBefore(
            load.getMemref(), before, inductionMemrefs, consumedOps)) {
      if (!llvm::is_contained(consumedOps, load.getOperation()))
        consumedOps.push_back(load.getOperation());
      return expression;
    }
    return {};
  }

  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumOperands() != 2)
    return {};

  ElementwiseIndexExprKind kind;
  if (isa<arith::AddIOp>(operation))
    kind = ElementwiseIndexExprKind::Add;
  else if (isa<arith::SubIOp>(operation))
    kind = ElementwiseIndexExprKind::Subtract;
  else if (isa<arith::MulIOp>(operation))
    kind = ElementwiseIndexExprKind::Multiply;
  else if (operation->getName().getStringRef() == "arith.minsi")
    kind = ElementwiseIndexExprKind::Min;
  else
    return {};

  auto lhs = matchGeneralIndexExpression(operation->getOperand(0),
                                         inductionMemrefs, before, consumedOps);
  auto rhs = matchGeneralIndexExpression(operation->getOperand(1),
                                         inductionMemrefs, before, consumedOps);
  if (!lhs || !rhs)
    return {};
  if (kind == ElementwiseIndexExprKind::Multiply &&
      getIndexExpressionLoopMask(lhs) != 0 &&
      getIndexExpressionLoopMask(rhs) != 0)
    return {};

  auto expression = makeIndexExpr(kind);
  expression->operands.push_back(std::move(lhs));
  expression->operands.push_back(std::move(rhs));
  return expression;
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

struct ScalarReferenceInfo {
  ScalarReferenceKind kind = ScalarReferenceKind::ReadOnlyCapture;
  Value reference;
  fir::LoadOp load;
  fir::StoreOp definingStore;
};

static bool scalarReferenceIsUsedAfterLaunch(Value reference,
                                             fir::fnacc::LaunchOp launchOp) {
  Block *launchBlock = launchOp->getBlock();

  for (Operation *user : reference.getUsers()) {
    if (launchOp->isProperAncestor(user))
      continue;

    // Find the operation containing this use that is directly nested in the
    // launch's block. This also catches uses inside a host-side fir.if after
    // the launch.
    Operation *blockOperation = user;
    while (blockOperation && blockOperation->getBlock() != launchBlock)
      blockOperation = blockOperation->getParentOp();

    if (blockOperation && blockOperation != launchOp.getOperation() &&
        launchOp->isBeforeInBlock(blockOperation))
      return true;
  }

  return false;
}

/// Classify a scalar load before it is admitted to the kernel ABI.
///
/// A reference with no stores in fnacc.launch is a read-only capture. A
/// reference with exactly one store in the same innermost loop block, before
/// every load, is an iteration-private temporary and can be promoted to SSA by
/// recursively recognizing the stored value. Everything else is explicitly
/// unsupported mutable state; in particular, loop-carried and conditionally
/// assigned scalars must not be silently captured as uniform kernel values.
static std::optional<ScalarReferenceInfo>
classifyScalarElementReference(Value v) {
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

  ScalarReferenceInfo info;
  info.reference = memref;
  info.load = load;

  auto launchOp = load->getParentOfType<fir::fnacc::LaunchOp>();
  if (!launchOp)
    return info;

  llvm::SmallVector<fir::StoreOp> stores;
  llvm::SmallVector<fir::LoadOp> loads;
  launchOp.walk([&](Operation *op) {
    if (auto candidateStore = dyn_cast<fir::StoreOp>(op)) {
      if (sameValueAfterFirConvert(candidateStore.getMemref(), memref))
        stores.push_back(candidateStore);
      return;
    }

    if (auto candidateLoad = dyn_cast<fir::LoadOp>(op))
      if (sameValueAfterFirConvert(candidateLoad.getMemref(), memref))
        loads.push_back(candidateLoad);
  });

  if (stores.empty())
    return info;

  info.kind = ScalarReferenceKind::UnsupportedMutable;
  if (stores.size() != 1)
    return info;

  // Private values have no copy-out semantics. Do not promote a scalar whose
  // host value is observed after the parallel region.
  if (scalarReferenceIsUsedAfterLaunch(memref, launchOp))
    return info;

  fir::StoreOp store = stores.front();
  fir::DoLoopOp storeLoop = store->getParentOfType<fir::DoLoopOp>();
  fir::DoLoopOp loadLoop = load->getParentOfType<fir::DoLoopOp>();

  // The store and every load must execute in the same logical iteration. The
  // same-block order requirement also rejects read-before-write and
  // loop-carried forms such as tmp = tmp + a(i).
  if (!storeLoop || storeLoop != loadLoop ||
      store->getBlock() != load->getBlock())
    return info;

  for (fir::LoadOp candidateLoad : loads) {
    if (candidateLoad->getBlock() != store->getBlock() ||
        !store->isBeforeInBlock(candidateLoad))
      return info;
  }

  info.kind = ScalarReferenceKind::IterationPrivate;
  info.definingStore = store;
  return info;
}

static std::optional<Value> getReadOnlyScalarCaptureRef(Value value) {
  std::optional<ScalarReferenceInfo> info =
      classifyScalarElementReference(value);
  if (!info || info->kind != ScalarReferenceKind::ReadOnlyCapture)
    return std::nullopt;

  return info->reference;
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

static unsigned getArrayRank(Value value) {
  Type type = unwrapArrayStorageType(value.getType());
  auto arrayType = dyn_cast<fir::SequenceType>(type);
  return arrayType ? arrayType.getDimension() : 0;
}

static bool inferAndCheckElementType(ElementwiseKernel &k,
                                     std::string &reason) {
  Value representativeWrite = k.writeArray;
  if (!representativeWrite && !k.writeArrays.empty())
    representativeWrite = k.writeArrays.front();

  if (!representativeWrite) {
    reason = "kernel has no write array";
    return false;
  }

  fir::fnacc::ElementType type = getArrayElementType(representativeWrite);
  if (type == fir::fnacc::ElementType::Unknown) {
    reason = "write array must be integer(1/2/4/8) or real(4/8)";
    return false;
  }

  for (Value read : k.readArrays) {
    fir::fnacc::ElementType readType = getArrayElementType(read);
    if (readType == fir::fnacc::ElementType::Unknown) {
      reason = "read arrays must be integer(1/2/4/8) or real(4/8)";
      return false;
    }
  }

  for (Value write : k.writeArrays) {
    if (getArrayElementType(write) != type) {
      reason = "all output arrays must have the same element type";
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
  llvm::SmallVector<llvm::SmallVector<unsigned, 3>> readDimensions;
  llvm::SmallVector<llvm::SmallVector<int64_t, 3>> readCoefficients;
  llvm::SmallVector<llvm::SmallVector<Value, 3>> readBaseRefs;
  llvm::SmallVector<llvm::SmallVector<int64_t, 3>> readOffsets;
  llvm::SmallVector<llvm::SmallVector<std::shared_ptr<ElementwiseIndexExpr>, 3>>
      readIndexExpressions;

  // A later store to one output may read the value written by an earlier
  // store to the same logical element. Such straight-line assignments are
  // fused into one expression root by forwarding the intervening load to the
  // earlier stored SSA value.
  llvm::SmallVector<std::pair<Value, Value>, 4> forwardedValues;

  // A fixed nested-loop expansion can visit one FIR load operation at several
  // logical stencil offsets. Expression recognition normally chooses the
  // first matching SSA load; these overrides select the occurrence belonging
  // to the currently unrolled iteration.
  llvm::SmallVector<std::pair<Value, unsigned>, 4> preferredReadIndices;

  Value writeArray;
  Value storedValue;
  llvm::SmallVector<Value> writeArrays;
  llvm::SmallVector<Value> storedValues;
  llvm::SmallVector<Operation *> writeStores;
  llvm::SmallVector<llvm::SmallVector<unsigned, 3>> writeDimensions;
  llvm::SmallVector<llvm::SmallVector<int64_t, 3>> writeCoefficients;
  llvm::SmallVector<llvm::SmallVector<Value, 3>> writeBaseRefs;
  llvm::SmallVector<llvm::SmallVector<int64_t, 3>> writeOffsets;
  llvm::SmallVector<llvm::SmallVector<std::shared_ptr<ElementwiseIndexExpr>, 3>>
      writeIndexExpressions;
  llvm::SmallVector<Operation *, 4> indexExpressionOps;
  struct WriteCondition {
    Operation *ifOperation = nullptr;
    Value condition;
    bool thenBranch = false;
  };
  llvm::SmallVector<llvm::SmallVector<WriteCondition, 2>> writeConditions;

  bool hasNonIdentitySubscript() const {
    auto hasOffset = [](ArrayRef<int64_t> offsets) {
      return llvm::any_of(offsets, [](int64_t value) { return value != 0; });
    };
    auto hasNonUnitCoefficient = [](ArrayRef<int64_t> coefficients) {
      return llvm::any_of(coefficients,
                          [](int64_t value) { return value != 1; });
    };
    auto hasBase = [](ArrayRef<Value> bases) {
      return llvm::any_of(bases, [](Value value) { return bool(value); });
    };
    auto hasGeneralExpression = [](const auto &expressions) {
      return llvm::any_of(expressions, [](const auto &expression) {
        return static_cast<bool>(expression);
      });
    };
    return llvm::any_of(readOffsets, hasOffset) ||
           llvm::any_of(writeOffsets, hasOffset) ||
           llvm::any_of(readCoefficients, hasNonUnitCoefficient) ||
           llvm::any_of(writeCoefficients, hasNonUnitCoefficient) ||
           llvm::any_of(readBaseRefs, hasBase) ||
           llvm::any_of(writeBaseRefs, hasBase) ||
           llvm::any_of(readIndexExpressions, hasGeneralExpression) ||
           llvm::any_of(writeIndexExpressions, hasGeneralExpression);
  }
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

static int findReadValueIndex(const ArrayAccessInfo &accesses, Value value) {
  Value stripped = stripFirConvert(value);
  for (const auto &[preferredValue, index] :
       llvm::reverse(accesses.preferredReadIndices)) {
    if (stripFirConvert(preferredValue) == stripped &&
        index < accesses.readValues.size())
      return static_cast<int>(index);
  }
  return findReadValueIndex(accesses.readValues, value);
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

/// Strip FIR operations that do not need separate expression nodes.
/// `fir.no_reassoc` records an optimization constraint. A representation-only
/// fir.convert between the same supported element type can also be discarded;
/// type-changing converts are retained and become explicit expression nodes.
static Value stripElementwiseWrappers(Value value) {
  while (true) {
    Operation *operation = value.getDefiningOp();
    if (auto convert = dyn_cast_or_null<fir::ConvertOp>(operation)) {
      ElementType sourceType =
          getSupportedElementType(convert.getValue().getType());
      ElementType resultType = getSupportedElementType(value.getType());
      if (sourceType != ElementType::Unknown && sourceType == resultType) {
        value = convert.getValue();
        continue;
      }
    }
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

static bool regionStoresScalarReference(Region &region, Value reference) {
  for (Block &block : region) {
    for (Operation &operation : block) {
      if (auto store = dyn_cast<fir::StoreOp>(operation))
        if (sameValueAfterFirConvert(store.getMemref(), reference))
          return true;
      for (Region &nested : operation.getRegions())
        if (regionStoresScalarReference(nested, reference))
          return true;
    }
  }
  return false;
}

/// Reconstruct the value of an iteration-private scalar immediately before an
/// operation. Structured fir.if assignments become expression-tree selects.
/// The walk never crosses the innermost loop boundary, so loop-carried state
/// remains unsupported.
static std::unique_ptr<ElementwiseExpr> recognizePrivateScalarValueBefore(
    Value reference, Operation *before, fir::DoLoopOp iterationLoop,
    const ArrayAccessInfo &accesses, ElementwiseKernel &kernel,
    std::string &reason) {
  Block *block = before->getBlock();
  for (auto iterator = before->getIterator(); iterator != block->begin();) {
    --iterator;
    Operation *operation = &*iterator;

    if (auto store = dyn_cast<fir::StoreOp>(operation)) {
      if (!sameValueAfterFirConvert(store.getMemref(), reference))
        continue;
      markConsumed(kernel, store.getOperation());
      return recognizeElementwiseExpr(store.getValue(), accesses, kernel,
                                      reason);
    }

    auto ifOp = dyn_cast<fir::IfOp>(operation);
    if (!ifOp)
      continue;
    bool thenStores =
        regionStoresScalarReference(ifOp.getThenRegion(), reference);
    bool elseStores =
        regionStoresScalarReference(ifOp.getElseRegion(), reference);
    if (!thenStores && !elseStores)
      continue;

    markConsumed(kernel, ifOp.getOperation());
    auto condition =
        recognizeElementwiseExpr(ifOp.getCondition(), accesses, kernel, reason);
    if (!condition)
      return nullptr;
    if (condition->resultKind != ElementwiseExprResultKind::Predicate) {
      reason = "iteration-private fir.if condition is not a predicate";
      return nullptr;
    }

    std::unique_ptr<ElementwiseExpr> trueValue;
    if (thenStores && !ifOp.getThenRegion().empty()) {
      Block &thenBlock = ifOp.getThenRegion().front();
      trueValue = recognizePrivateScalarValueBefore(
          reference, thenBlock.getTerminator(), iterationLoop, accesses, kernel,
          reason);
    } else {
      trueValue = recognizePrivateScalarValueBefore(
          reference, ifOp.getOperation(), iterationLoop, accesses, kernel,
          reason);
    }
    if (!trueValue)
      return nullptr;

    std::unique_ptr<ElementwiseExpr> falseValue;
    if (elseStores && !ifOp.getElseRegion().empty()) {
      Block &elseBlock = ifOp.getElseRegion().front();
      falseValue = recognizePrivateScalarValueBefore(
          reference, elseBlock.getTerminator(), iterationLoop, accesses, kernel,
          reason);
    } else {
      falseValue = recognizePrivateScalarValueBefore(
          reference, ifOp.getOperation(), iterationLoop, accesses, kernel,
          reason);
    }
    if (!falseValue)
      return nullptr;

    auto expression = makeExpr(ElementwiseExprKind::Select);
    expression->elementType = trueValue->elementType;
    expression->operands.push_back(std::move(condition));
    expression->operands.push_back(std::move(trueValue));
    expression->operands.push_back(std::move(falseValue));
    return expression;
  }

  Operation *parent = block->getParentOp();
  if (!parent || parent == iterationLoop.getOperation() ||
      isa<fir::fnacc::LaunchOp>(parent)) {
    reason = "mutable scalar reference is neither iteration-private nor a "
             "reduction";
    return nullptr;
  }

  return recognizePrivateScalarValueBefore(reference, parent, iterationLoop,
                                           accesses, kernel, reason);
}

static bool matchFixedTwoIterationLoop(fir::DoLoopOp loop,
                                       Value coordinateMemref) {
  if (!loop || !isConstantIntegerValue(loop.getStep(), 1))
    return false;

  std::optional<AffineIndexMatch> lower =
      matchAffineIndex(loop.getLowerBound(), coordinateMemref);
  std::optional<AffineIndexMatch> upper =
      matchAffineIndex(loop.getUpperBound(), coordinateMemref);
  return lower && upper && lower->coefficient == 1 && upper->coefficient == 1 &&
         !lower->baseRef && !upper->baseRef && lower->offset == 0 &&
         upper->offset == 1;
}

/// Expand the small vertex average used by CloverLeaf field-summary kernels:
///
///   acc = 0
///   do kv = k, k + 1
///     do jv = j, j + 1
///       acc = acc + term(jv, kv)
///     end do
///   end do
///
/// The array collector records the one FIR load at each of the four logical
/// offsets. This routine selects the appropriate occurrence while rebuilding
/// the accumulator as a straight-line expression tree.
static std::unique_ptr<ElementwiseExpr>
recognizeFixedNestedScalarSum(fir::LoadOp load, const ArrayAccessInfo &accesses,
                              ElementwiseKernel &kernel, std::string &reason) {
  Value reference = load.getMemref();
  fir::DoLoopOp innerLogicalLoop = load->getParentOfType<fir::DoLoopOp>();
  if (!innerLogicalLoop || load->getBlock() != innerLogicalLoop.getBody())
    return nullptr;

  fir::DoLoopOp outerLogicalLoop =
      innerLogicalLoop->getParentOfType<fir::DoLoopOp>();
  if (!outerLogicalLoop)
    return nullptr;

  Value logicalX = findInductionMemref(innerLogicalLoop);
  Value logicalY = findInductionMemref(outerLogicalLoop);
  if (!logicalX || !logicalY)
    return nullptr;

  fir::StoreOp initialization;
  fir::DoLoopOp vertexYLoop;
  for (Operation &operation : load->getBlock()->getOperations()) {
    if (&operation == load.getOperation())
      break;
    if (auto store = dyn_cast<fir::StoreOp>(operation)) {
      if (sameValueAfterFirConvert(store.getMemref(), reference))
        initialization = store;
      continue;
    }
    auto loop = dyn_cast<fir::DoLoopOp>(operation);
    if (!loop || !regionStoresScalarReference(loop.getRegion(), reference))
      continue;
    if (vertexYLoop) {
      reason = "private scalar has more than one fixed nested update loop";
      return nullptr;
    }
    vertexYLoop = loop;
  }

  if (!initialization || !vertexYLoop)
    return nullptr;
  if (!initialization->isBeforeInBlock(vertexYLoop.getOperation()) ||
      !matchFixedTwoIterationLoop(vertexYLoop, logicalY)) {
    reason = "private scalar nested sum requires an initialization followed "
             "by k:k+1";
    return nullptr;
  }

  fir::DoLoopOp vertexXLoop;
  for (Operation &operation : vertexYLoop.getBody()->getOperations()) {
    auto loop = dyn_cast<fir::DoLoopOp>(operation);
    if (!loop)
      continue;
    if (vertexXLoop) {
      reason = "private scalar nested sum has more than two loop levels";
      return nullptr;
    }
    vertexXLoop = loop;
  }
  if (!vertexXLoop || !matchFixedTwoIterationLoop(vertexXLoop, logicalX)) {
    reason = "private scalar nested sum requires an inner j:j+1 loop";
    return nullptr;
  }

  fir::StoreOp update;
  for (Operation &operation : vertexXLoop.getBody()->getOperations()) {
    auto store = dyn_cast<fir::StoreOp>(operation);
    if (!store || !sameValueAfterFirConvert(store.getMemref(), reference))
      continue;
    if (update) {
      reason = "private scalar nested sum has multiple accumulator stores";
      return nullptr;
    }
    update = store;
  }
  if (!update)
    return nullptr;

  Value updateValue = stripElementwiseWrappers(update.getValue());
  Operation *add = updateValue.getDefiningOp();
  if (!add ||
      (add->getName().getStringRef() != "arith.addf" &&
       add->getName().getStringRef() != "arith.addi") ||
      add->getNumOperands() != 2) {
    reason = "private scalar nested sum update is not an addition";
    return nullptr;
  }

  auto isAccumulatorLoad = [&](Value value) {
    value = stripFirConvert(value);
    auto candidate = value.getDefiningOp<fir::LoadOp>();
    return candidate &&
           sameValueAfterFirConvert(candidate.getMemref(), reference);
  };
  Value term;
  if (isAccumulatorLoad(add->getOperand(0)))
    term = add->getOperand(1);
  else if (isAccumulatorLoad(add->getOperand(1)))
    term = add->getOperand(0);
  else {
    reason = "private scalar nested sum does not read its previous value";
    return nullptr;
  }

  markConsumed(kernel, initialization.getOperation());
  markConsumed(kernel, update.getOperation());
  getOrAddValueIndex(kernel.privateScalarRefs, reference);

  std::unique_ptr<ElementwiseExpr> accumulated = recognizeElementwiseExpr(
      initialization.getValue(), accesses, kernel, reason);
  if (!accumulated)
    return nullptr;

  for (int64_t yOffset = 0; yOffset != 2; ++yOffset) {
    for (int64_t xOffset = 0; xOffset != 2; ++xOffset) {
      ArrayAccessInfo iterationAccesses = accesses;
      bool selectedNestedAccess = false;
      for (unsigned index = 0; index < accesses.readValues.size(); ++index) {
        Operation *read =
            stripFirConvert(accesses.readValues[index]).getDefiningOp();
        if (!read || !vertexXLoop->isProperAncestor(read))
          continue;
        if (index >= accesses.readDimensions.size() ||
            index >= accesses.readOffsets.size())
          continue;

        bool matches = true;
        for (auto [dimension, offset] : llvm::zip_equal(
                 accesses.readDimensions[index], accesses.readOffsets[index])) {
          int64_t expected = dimension == 0 ? xOffset : yOffset;
          if (offset != expected) {
            matches = false;
            break;
          }
        }
        if (!matches)
          continue;
        iterationAccesses.preferredReadIndices.push_back(
            {accesses.readValues[index], index});
        selectedNestedAccess = true;
      }
      if (!selectedNestedAccess) {
        reason = "private scalar nested sum has no array access for one "
                 "unrolled iteration";
        return nullptr;
      }

      std::unique_ptr<ElementwiseExpr> termExpression =
          recognizeElementwiseExpr(term, iterationAccesses, kernel, reason);
      if (!termExpression)
        return nullptr;

      auto sum = makeExpr(add->getName().getStringRef() == "arith.addf"
                              ? ElementwiseExprKind::AddF
                              : ElementwiseExprKind::AddI);
      sum->elementType = accumulated->elementType;
      sum->operands.push_back(std::move(accumulated));
      sum->operands.push_back(std::move(termExpression));
      accumulated = std::move(sum);
    }
  }
  return accumulated;
}

static std::unique_ptr<ElementwiseExpr> recognizeStructuredPrivateScalarLoad(
    fir::LoadOp load, const ArrayAccessInfo &accesses,
    ElementwiseKernel &kernel, std::string &reason) {
  Value reference = load.getMemref();
  auto launchOp = load->getParentOfType<fir::fnacc::LaunchOp>();
  fir::DoLoopOp iterationLoop = load->getParentOfType<fir::DoLoopOp>();
  if (!launchOp || !iterationLoop) {
    reason = "mutable scalar load is outside a logical loop iteration";
    return nullptr;
  }

  std::string fixedNestedReason;
  if (std::unique_ptr<ElementwiseExpr> fixedNested =
          recognizeFixedNestedScalarSum(load, accesses, kernel,
                                        fixedNestedReason))
    return fixedNested;
  if (!fixedNestedReason.empty()) {
    reason = std::move(fixedNestedReason);
    return nullptr;
  }

  if (scalarReferenceIsUsedAfterLaunch(reference, launchOp)) {
    reason = "conditionally assigned scalar is observed after the launch";
    return nullptr;
  }

  bool hasStore = false;
  bool invalidStore = false;
  launchOp.walk([&](fir::StoreOp store) {
    if (!sameValueAfterFirConvert(store.getMemref(), reference))
      return;
    hasStore = true;
    fir::DoLoopOp storeLoop = store->getParentOfType<fir::DoLoopOp>();
    if (!storeLoop || storeLoop.getOperation() != iterationLoop.getOperation())
      invalidStore = true;
  });
  if (!hasStore || invalidStore) {
    reason = "conditional scalar assignments do not belong to one iteration";
    return nullptr;
  }

  getOrAddValueIndex(kernel.privateScalarRefs, reference);
  markConsumed(kernel, load.getOperation());
  return recognizePrivateScalarValueBefore(
      reference, load.getOperation(), iterationLoop, accesses, kernel, reason);
}

static std::unique_ptr<ElementwiseExpr>
recognizeUnaryElementwiseExpr(ElementwiseExprKind kind, Value operand,
                              const ArrayAccessInfo &accesses,
                              ElementwiseKernel &kernel, std::string &reason) {
  auto child = recognizeElementwiseExpr(operand, accesses, kernel, reason);
  if (!child)
    return nullptr;

  auto expression = makeExpr(kind);
  expression->elementType = child->elementType;
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
  expression->elementType = lhs->elementType;
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

  // Preserve a type-changing conversion before any lookup that deliberately
  // peers through fir.convert (array loads, forwarded values, and affine
  // indices). Otherwise the conversion is mistaken for its source value and
  // silently disappears from the emitted expression.
  if (auto convert = value.getDefiningOp<fir::ConvertOp>()) {
    ElementType sourceType =
        getSupportedElementType(convert.getValue().getType());
    ElementType destinationType = getSupportedElementType(value.getType());
    if (sourceType == ElementType::Unknown ||
        destinationType == ElementType::Unknown) {
      reason = "fir.convert uses an unsupported element type";
      return nullptr;
    }
    auto child =
        recognizeElementwiseExpr(convert.getValue(), accesses, kernel, reason);
    if (!child)
      return nullptr;
    auto expression = makeExpr(ElementwiseExprKind::Convert);
    expression->elementType = destinationType;
    expression->operands.push_back(std::move(child));
    return expression;
  }

  // Resolve an iteration-local read-after-write before treating the load as
  // a device-memory input. Reverse iteration selects the most recent
  // forwarding definition if a chain contains more than two stores.
  for (const auto &forwarded : llvm::reverse(accesses.forwardedValues)) {
    if (!sameValueAfterFirConvert(value, forwarded.first))
      continue;
    return recognizeElementwiseExpr(forwarded.second, accesses, kernel, reason);
  }

  // Array element load previously collected from a recognized array_coor.
  int readIndex = findReadValueIndex(accesses, value);
  if (readIndex >= 0) {
    Value arrayBase = accesses.readArrays[readIndex];
    unsigned arrayIndex = getOrAddArrayIndex(kernel.readArrays, arrayBase);

    auto expression = makeExpr(ElementwiseExprKind::ArrayLoad);
    expression->source = kernel.readArrays[arrayIndex];
    expression->arrayAccessIndex = readIndex;
    expression->elementType = getArrayElementType(arrayBase);
    return expression;
  }

  // An induction-derived value may participate in the stored expression as
  // well as in an array address. Preserve the source Fortran coordinate so a
  // non-unit loop origin does not become the zero-based device offset. The
  // captured integer base is kept separate from element-typed scalar captures.
  if (kernel.innerIndMemref) {
    std::optional<AffineIndexMatch> affine =
        matchAffineIndexExpression(value, kernel.innerIndMemref);
    if (affine && (affine->coefficient == 1 || affine->coefficient == -1) &&
        (!affine->baseRef || affine->baseCoefficient == 1 ||
         affine->baseCoefficient == -1)) {
      auto expression = makeExpr(ElementwiseExprKind::AffineIndex);
      expression->elementType = getSupportedElementType(value.getType());
      expression->affineCoefficient = affine->coefficient;
      expression->affineBaseCoefficient = affine->baseCoefficient;
      expression->affineOffset = affine->offset;
      if (affine->baseRef)
        expression->affineBaseIndex = static_cast<int32_t>(
            getOrAddValueIndex(kernel.indexRefs, affine->baseRef));
      return expression;
    }
  }

  // Classify scalar storage before deciding whether it belongs in the ABI.
  if (auto scalar = classifyScalarElementReference(value)) {
    switch (scalar->kind) {
    case ScalarReferenceKind::ReadOnlyCapture: {
      ElementType scalarType = getScalarRefElementType(scalar->reference);
      bool useIndexBinding = kernel.elementType != ElementType::Unknown &&
                             scalarType != kernel.elementType &&
                             scalarType == ElementType::I32;
      getOrAddValueIndex(useIndexBinding ? kernel.indexRefs : kernel.scalarRefs,
                         scalar->reference);

      auto expression =
          makeExpr(useIndexBinding ? ElementwiseExprKind::IndexScalarLoad
                                   : ElementwiseExprKind::ScalarLoad);
      expression->source = scalar->reference;
      expression->elementType = scalarType;
      return expression;
    }

    case ScalarReferenceKind::IterationPrivate: {
      getOrAddValueIndex(kernel.privateScalarRefs, scalar->reference);
      markConsumed(kernel, scalar->load.getOperation());
      markConsumed(kernel, scalar->definingStore.getOperation());

      auto expression = recognizeElementwiseExpr(
          scalar->definingStore.getValue(), accesses, kernel, reason);
      if (!expression && reason.empty())
        reason = "iteration-private scalar has unsupported defining value";
      return expression;
    }

    case ScalarReferenceKind::UnsupportedMutable:
      return recognizeStructuredPrivateScalarLoad(scalar->load, accesses,
                                                  kernel, reason);
    }
  }

  // Floating-point constant.
  if (auto constantValue = getRealConstantValue(value)) {
    auto expression = makeExpr(ElementwiseExprKind::ConstantReal);
    expression->realValue = *constantValue;
    expression->elementType = getSupportedElementType(value.getType());
    return expression;
  }

  if (auto constantValue = getIntegerConstantValue(value)) {
    auto expression = makeExpr(ElementwiseExprKind::ConstantInteger);
    expression->integerValue = *constantValue;
    expression->elementType = getSupportedElementType(value.getType());
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

  if (operation->getName().getStringRef() == "math.fpowi" &&
      operation->getNumOperands() == 2) {
    std::optional<int64_t> exponent =
        getIntegerConstantValue(operation->getOperand(1));
    if (!exponent || *exponent != 2) {
      reason = "elementwise integer power currently requires exponent 2";
      return nullptr;
    }
    return recognizeUnaryElementwiseExpr(ElementwiseExprKind::SquareF,
                                         operation->getOperand(0), accesses,
                                         kernel, reason);
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

  if ((operationName == "arith.andi" || operationName == "arith.ori") &&
      operation->getNumOperands() == 2 && resultType.isInteger(1)) {
    ElementwiseExprKind kind = operationName == "arith.andi"
                                   ? ElementwiseExprKind::And
                                   : ElementwiseExprKind::Or;
    return recognizeBinaryElementwiseExpr(
        kind, operation->getOperand(0), operation->getOperand(1), accesses,
        kernel, reason, ElementwiseExprResultKind::Predicate);
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
    expression->elementType = trueValue->elementType;
    expression->operands.push_back(std::move(condition));
    expression->operands.push_back(std::move(trueValue));
    expression->operands.push_back(std::move(falseValue));
    return expression;
  }

  reason = "unsupported operation in elementwise expression tree: ";
  reason += operationName;
  return nullptr;
}

static unsigned getOrAddArrayArgument(ElementwiseKernel &kernel, Value array,
                                      bool read, bool write);

static llvm::SmallVector<int32_t, 3>
getAffineBaseIndices(ElementwiseKernel &kernel, ArrayRef<Value> baseRefs) {
  llvm::SmallVector<int32_t, 3> indices;
  indices.reserve(baseRefs.size());
  for (Value baseRef : baseRefs) {
    if (!baseRef) {
      indices.push_back(-1);
      continue;
    }
    indices.push_back(
        static_cast<int32_t>(getOrAddValueIndex(kernel.indexRefs, baseRef)));
  }
  return indices;
}

static void bindIndexExpressionCaptures(
    ElementwiseKernel &kernel,
    const std::shared_ptr<ElementwiseIndexExpr> &expression) {
  if (!expression)
    return;

  if (expression->kind == ElementwiseIndexExprKind::Capture) {
    expression->captureIndex = static_cast<int32_t>(
        getOrAddValueIndex(kernel.indexRefs, expression->capture));
  }

  for (const auto &operand : expression->operands)
    bindIndexExpressionCaptures(kernel, operand);
}

static void bindIndexExpressionCaptures(
    ElementwiseKernel &kernel,
    ArrayRef<std::shared_ptr<ElementwiseIndexExpr>> expressions) {
  for (const auto &expression : expressions)
    bindIndexExpressionCaptures(kernel, expression);
}

static bool bindIndexExpressionConditions(
    ElementwiseKernel &kernel,
    const std::shared_ptr<ElementwiseIndexExpr> &expression,
    const ArrayAccessInfo &accesses, std::string &reason) {
  if (!expression)
    return true;

  if (expression->kind == ElementwiseIndexExprKind::Select) {
    std::unique_ptr<ElementwiseExpr> condition = recognizeElementwiseExpr(
        expression->condition, accesses, kernel, reason);
    if (!condition)
      return false;
    if (condition->resultKind != ElementwiseExprResultKind::Predicate) {
      reason = "conditional stencil index condition is not a predicate";
      return false;
    }
    expression->conditionExpression =
        std::shared_ptr<ElementwiseExpr>(std::move(condition));
  }

  for (const auto &operand : expression->operands)
    if (!bindIndexExpressionConditions(kernel, operand, accesses, reason))
      return false;
  return true;
}

static bool bindIndexExpressionConditions(
    ElementwiseKernel &kernel,
    ArrayRef<std::shared_ptr<ElementwiseIndexExpr>> expressions,
    const ArrayAccessInfo &accesses, std::string &reason) {
  for (const auto &expression : expressions)
    if (!bindIndexExpressionConditions(kernel, expression, accesses, reason))
      return false;
  return true;
}

static std::unique_ptr<ElementwiseExpr>
buildWritePredicate(ArrayRef<ArrayAccessInfo::WriteCondition> conditions,
                    const ArrayAccessInfo &info, ElementwiseKernel &kernel,
                    std::string &reason) {
  std::unique_ptr<ElementwiseExpr> predicate;
  for (const auto &writeCondition : conditions) {
    markConsumed(kernel, writeCondition.ifOperation);
    auto condition = recognizeElementwiseExpr(writeCondition.condition, info,
                                              kernel, reason);
    if (!condition)
      return nullptr;
    if (condition->resultKind != ElementwiseExprResultKind::Predicate) {
      reason = "guarded output condition is not a predicate";
      return nullptr;
    }

    if (!writeCondition.thenBranch) {
      auto negated = makeExpr(ElementwiseExprKind::Not);
      negated->resultKind = ElementwiseExprResultKind::Predicate;
      negated->elementType = condition->elementType;
      negated->operands.push_back(std::move(condition));
      condition = std::move(negated);
    }

    if (!predicate) {
      predicate = std::move(condition);
      continue;
    }

    auto conjunction = makeExpr(ElementwiseExprKind::And);
    conjunction->resultKind = ElementwiseExprResultKind::Predicate;
    conjunction->elementType = predicate->elementType;
    conjunction->operands.push_back(std::move(predicate));
    conjunction->operands.push_back(std::move(condition));
    predicate = std::move(conjunction);
  }
  return predicate;
}

static bool appendRecognizedOutputs(ElementwiseKernel &kernel,
                                    const ArrayAccessInfo &info,
                                    std::string &reason) {
  auto makeOutput = [&](unsigned index) {
    ElementwiseOutput output;
    output.array = info.writeArrays[index];
    output.storedValue = info.storedValues[index];
    output.arrayArgumentIndex =
        getOrAddArrayArgument(kernel, output.array, false, true);
    output.dimensions = info.writeDimensions[index];
    output.coefficients = info.writeCoefficients[index];
    output.baseIndices =
        getAffineBaseIndices(kernel, info.writeBaseRefs[index]);
    output.offsets = info.writeOffsets[index];
    output.indexExpressions = info.writeIndexExpressions[index];
    bindIndexExpressionCaptures(kernel, output.indexExpressions);
    return output;
  };

  llvm::SmallVector<bool> handled(info.storedValues.size(), false);
  for (unsigned first = 0; first < info.storedValues.size(); ++first) {
    if (handled[first])
      continue;

    llvm::SmallVector<unsigned, 2> group;
    for (unsigned candidate = first; candidate < info.storedValues.size();
         ++candidate) {
      if (handled[candidate] ||
          !sameArrayBase(info.writeArrays[first],
                         info.writeArrays[candidate]) ||
          info.writeDimensions[first] != info.writeDimensions[candidate] ||
          info.writeCoefficients[first] != info.writeCoefficients[candidate] ||
          info.writeBaseRefs[first] != info.writeBaseRefs[candidate] ||
          info.writeOffsets[first] != info.writeOffsets[candidate] ||
          !sameIndexExpressions(info.writeIndexExpressions[first],
                                info.writeIndexExpressions[candidate]))
        continue;
      handled[candidate] = true;
      group.push_back(candidate);
    }
    ElementwiseOutput output = makeOutput(first);

    bool allUnconditional = llvm::all_of(group, [&](unsigned index) {
      return info.writeConditions[index].empty();
    });
    if (allUnconditional) {
      ArrayAccessInfo forwardedInfo = info;
      for (unsigned stage = 1; stage < group.size(); ++stage) {
        unsigned previousIndex = group[stage - 1];
        unsigned currentIndex = group[stage];
        Operation *previousStore = info.writeStores[previousIndex];
        Operation *currentStore = info.writeStores[currentIndex];
        bool foundForwardedLoad = false;

        if (!previousStore || !currentStore ||
            previousStore->getBlock() != currentStore->getBlock()) {
          reason = "unconditional output store chain crosses a block boundary";
          return false;
        }

        for (unsigned readIndex = 0; readIndex < info.readValues.size();
             ++readIndex) {
          if (!sameArrayBase(info.readArrays[readIndex], output.array) ||
              info.readDimensions[readIndex] != output.dimensions ||
              info.readCoefficients[readIndex] != output.coefficients ||
              info.readBaseRefs[readIndex] != info.writeBaseRefs[first] ||
              info.readOffsets[readIndex] != output.offsets ||
              !sameIndexExpressions(info.readIndexExpressions[readIndex],
                                    output.indexExpressions))
            continue;

          Operation *load = info.readValues[readIndex].getDefiningOp();
          if (!load || load->getBlock() != previousStore->getBlock() ||
              !previousStore->isBeforeInBlock(load) ||
              !load->isBeforeInBlock(currentStore))
            continue;

          forwardedInfo.forwardedValues.push_back(
              {info.readValues[readIndex], info.storedValues[previousIndex]});
          foundForwardedLoad = true;
        }

        if (!foundForwardedLoad) {
          reason = "multiple unconditional output stores are not a "
                   "same-element read-after-write chain";
          return false;
        }
      }

      unsigned finalIndex = group.back();
      output.storedValue = info.storedValues[finalIndex];
      output.expression = recognizeElementwiseExpr(
          output.storedValue, forwardedInfo, kernel, reason);
    } else if (group.size() == 2 &&
               info.writeConditions[group[0]].size() == 1 &&
               info.writeConditions[group[1]].size() == 1 &&
               info.writeConditions[group[0]].front().ifOperation ==
                   info.writeConditions[group[1]].front().ifOperation &&
               info.writeConditions[group[0]].front().condition ==
                   info.writeConditions[group[1]].front().condition &&
               info.writeConditions[group[0]].front().thenBranch !=
                   info.writeConditions[group[1]].front().thenBranch) {
      const auto &lhs = info.writeConditions[group[0]].front();

      unsigned trueIndex = lhs.thenBranch ? group[0] : group[1];
      unsigned falseIndex = lhs.thenBranch ? group[1] : group[0];
      markConsumed(kernel, lhs.ifOperation);

      auto condition =
          recognizeElementwiseExpr(lhs.condition, info, kernel, reason);
      if (!condition)
        return false;
      if (condition->resultKind != ElementwiseExprResultKind::Predicate) {
        reason = "conditional output condition is not a predicate";
        return false;
      }

      auto trueValue = recognizeElementwiseExpr(info.storedValues[trueIndex],
                                                info, kernel, reason);
      if (!trueValue)
        return false;
      auto falseValue = recognizeElementwiseExpr(info.storedValues[falseIndex],
                                                 info, kernel, reason);
      if (!falseValue)
        return false;

      output.storedValue = info.storedValues[trueIndex];
      output.expression = makeExpr(ElementwiseExprKind::Select);
      output.expression->elementType = trueValue->elementType;
      output.expression->operands.push_back(std::move(condition));
      output.expression->operands.push_back(std::move(trueValue));
      output.expression->operands.push_back(std::move(falseValue));
    } else {
      for (unsigned index : group) {
        ElementwiseOutput guarded = makeOutput(index);
        guarded.expression =
            recognizeElementwiseExpr(guarded.storedValue, info, kernel, reason);
        if (!guarded.expression)
          return false;
        guarded.predicate = buildWritePredicate(info.writeConditions[index],
                                                info, kernel, reason);
        if (!guarded.predicate)
          return false;
        kernel.outputs.push_back(std::move(guarded));
      }
      continue;
    }

    if (!output.expression)
      return false;
    kernel.outputs.push_back(std::move(output));
  }
  return true;
}

static bool detectGenericExpr(ElementwiseKernel &k, const ArrayAccessInfo &info,
                              int rank, std::string &reason) {
  k.kind =
      rank == 1 ? ElementwiseKernelKind::Expr1D : ElementwiseKernelKind::Expr2D;
  k.rank = rank;
  k.writeArray = info.writeArray;
  k.readArrays.clear();
  k.scalarRefs.clear();
  k.privateScalarRefs.clear();
  k.computeOp = info.storedValue.getDefiningOp();

  auto expr = recognizeElementwiseExpr(info.storedValue, info, k, reason);
  if (!expr)
    return false;

  k.expression = std::move(expr);

  if (!inferAndCheckElementType(k, reason))
    return false;

  return true;
}

static unsigned getOrAddArrayArgument(ElementwiseKernel &kernel, Value array,
                                      bool read, bool write) {
  unsigned rank = getArrayRank(array);
  ElementType elementType = getArrayElementType(array);
  for (auto [index, argument] : llvm::enumerate(kernel.arrayArguments)) {
    if (!sameArrayBase(argument.array, array))
      continue;
    argument.read |= read;
    argument.write |= write;
    if (argument.rank == 0)
      argument.rank = rank;
    if (argument.elementType == ElementType::Unknown)
      argument.elementType = elementType;
    return index;
  }

  ElementwiseArrayArgument argument;
  argument.array = array;
  argument.read = read;
  argument.write = write;
  argument.rank = rank;
  argument.elementType = elementType;
  kernel.arrayArguments.push_back(argument);
  return kernel.arrayArguments.size() - 1;
}

static bool detectMultiExpr1D(ElementwiseKernel &kernel,
                              const ArrayAccessInfo &info,
                              std::string &reason) {
  kernel.kind = ElementwiseKernelKind::MultiExpr1D;
  kernel.rank = 1;
  kernel.readArrays.clear();
  kernel.writeArrays.clear();
  kernel.arrayArguments.clear();
  kernel.arrayAccesses.clear();
  kernel.outputs.clear();
  kernel.scalarRefs.clear();
  kernel.indexRefs.clear();
  kernel.privateScalarRefs.clear();
  kernel.elementType = getArrayElementType(info.writeArray);

  for (Operation *op : info.indexExpressionOps)
    markConsumed(kernel, op);

  for (unsigned index = 0; index < info.readValues.size(); ++index) {
    Value array = info.readArrays[index];
    getOrAddArrayIndex(kernel.readArrays, array);
    unsigned arrayArgumentIndex =
        getOrAddArrayArgument(kernel, array, true, false);

    ElementwiseArrayAccess access;
    access.array = array;
    access.loadedValue = info.readValues[index];
    access.arrayArgumentIndex = arrayArgumentIndex;
    access.elementType = getArrayElementType(array);
    access.dimensions = info.readDimensions[index];
    access.coefficients = info.readCoefficients[index];
    access.baseIndices = getAffineBaseIndices(kernel, info.readBaseRefs[index]);
    access.offsets = info.readOffsets[index];
    access.indexExpressions = info.readIndexExpressions[index];
    bindIndexExpressionCaptures(kernel, access.indexExpressions);
    kernel.arrayAccesses.push_back(std::move(access));
  }

  for (ElementwiseArrayAccess &access : kernel.arrayAccesses)
    if (!bindIndexExpressionConditions(kernel, access.indexExpressions, info,
                                       reason))
      return false;

  for (Value array : info.writeArrays) {
    getOrAddArrayIndex(kernel.writeArrays, array);
    getOrAddArrayArgument(kernel, array, false, true);
  }
  if (kernel.writeArrays.empty()) {
    reason = "multi-expression kernel has no output arrays";
    return false;
  }
  kernel.writeArray = kernel.writeArrays.front();

  if (!appendRecognizedOutputs(kernel, info, reason))
    return false;
  for (ElementwiseOutput &output : kernel.outputs)
    if (!bindIndexExpressionConditions(kernel, output.indexExpressions, info,
                                       reason))
      return false;

  kernel.computeOp = kernel.outputs.front().storedValue.getDefiningOp();
  return inferAndCheckElementType(kernel, reason);
}

static bool detectStencil2D(ElementwiseKernel &kernel,
                            const ArrayAccessInfo &info, std::string &reason) {
  kernel.kind = ElementwiseKernelKind::Stencil2D;
  kernel.rank = 2;
  kernel.readArrays.clear();
  kernel.writeArrays.clear();
  kernel.arrayArguments.clear();
  kernel.arrayAccesses.clear();
  kernel.outputs.clear();
  kernel.scalarRefs.clear();
  kernel.indexRefs.clear();
  kernel.privateScalarRefs.clear();
  kernel.elementType = getArrayElementType(info.writeArray);

  for (Operation *op : info.indexExpressionOps)
    markConsumed(kernel, op);

  for (unsigned index = 0; index < info.readValues.size(); ++index) {
    Value array = info.readArrays[index];
    getOrAddArrayIndex(kernel.readArrays, array);
    unsigned arrayArgumentIndex =
        getOrAddArrayArgument(kernel, array, true, false);

    ElementwiseArrayAccess access;
    access.array = array;
    access.loadedValue = info.readValues[index];
    access.arrayArgumentIndex = arrayArgumentIndex;
    access.elementType = getArrayElementType(array);
    access.dimensions = info.readDimensions[index];
    access.coefficients = info.readCoefficients[index];
    access.baseIndices = getAffineBaseIndices(kernel, info.readBaseRefs[index]);
    access.offsets = info.readOffsets[index];
    access.indexExpressions = info.readIndexExpressions[index];
    bindIndexExpressionCaptures(kernel, access.indexExpressions);
    kernel.arrayAccesses.push_back(std::move(access));
  }

  for (ElementwiseArrayAccess &access : kernel.arrayAccesses)
    if (!bindIndexExpressionConditions(kernel, access.indexExpressions, info,
                                       reason))
      return false;

  for (Value array : info.writeArrays) {
    getOrAddArrayIndex(kernel.writeArrays, array);
    getOrAddArrayArgument(kernel, array, false, true);
  }

  if (kernel.writeArrays.empty()) {
    reason = "stencil kernel has no output arrays";
    return false;
  }
  kernel.writeArray = kernel.writeArrays.front();

  if (!appendRecognizedOutputs(kernel, info, reason))
    return false;
  for (ElementwiseOutput &output : kernel.outputs)
    if (!bindIndexExpressionConditions(kernel, output.indexExpressions, info,
                                       reason))
      return false;

  kernel.computeOp = kernel.outputs.front().storedValue.getDefiningOp();
  return inferAndCheckElementType(kernel, reason);
}

static bool collectArrayAccessesFromBody(
    Block *body, unsigned expectedRank, ArrayRef<Value> expectedIndexMemrefs,
    ArrayAccessInfo &info, std::string &reason, unsigned minReads = 1,
    unsigned maxReads = 3, bool allowMultipleWrites = false,
    bool allowAffineSubscripts = false, bool requireWriteArray = true) {
  if (!body) {
    reason = "loop has no body";
    return false;
  }

  using WriteCondition = ArrayAccessInfo::WriteCondition;
  struct InductionSubstitution {
    Value memref;
    unsigned dimension = 0;
    int64_t offset = 0;
  };
  using WriteConditions = llvm::SmallVector<WriteCondition, 4>;
  using InductionSubstitutions = llvm::SmallVector<InductionSubstitution, 2>;
  std::function<bool(Block *, WriteConditions, InductionSubstitutions)>
      collectBlock;
  collectBlock = [&](Block *current, WriteConditions conditions,
                     InductionSubstitutions substitutions) {
    auto recordStructuralOp = [&](Operation *op) {
      if (op && !llvm::is_contained(info.indexExpressionOps, op))
        info.indexExpressionOps.push_back(op);
    };

    auto matchSubstitutedInduction = [&](Value value)
        -> std::optional<std::pair<unsigned, AffineIndexMatch>> {
      value = stripIndexExpressionWrappers(value);
      auto load = value.getDefiningOp<fir::LoadOp>();
      if (!load)
        return std::nullopt;
      for (const auto &substitution : substitutions) {
        if (!sameValueAfterFirConvert(load.getMemref(), substitution.memref))
          continue;
        AffineIndexMatch match;
        match.coefficient = 1;
        match.offset = substitution.offset;
        return std::pair<unsigned, AffineIndexMatch>{substitution.dimension,
                                                     match};
      }
      return std::nullopt;
    };

    auto matchLogicalCoordinate = [&](Value value)
        -> std::optional<std::pair<unsigned, AffineIndexMatch>> {
      std::optional<std::pair<unsigned, AffineIndexMatch>> result;
      for (unsigned dimension = 0; dimension < expectedRank; ++dimension) {
        auto match = matchAffineIndex(value, expectedIndexMemrefs[dimension]);
        if (!match)
          continue;
        if (result)
          return std::nullopt;
        result = std::pair<unsigned, AffineIndexMatch>{dimension, *match};
      }
      if (result)
        return result;
      return matchSubstitutedInduction(value);
    };

    for (Operation &operation : current->getOperations()) {
      if (auto arrayCoor = dyn_cast<fir::ArrayCoorOp>(operation)) {
        auto indices = arrayCoor.getIndices();
        if (indices.empty() || indices.size() > expectedRank ||
            (expectedRank == 1 && indices.size() != 1)) {
          reason = "array_coor rank is incompatible with kernel rank";
          return false;
        }

        llvm::SmallVector<unsigned, 3> dimensions;
        llvm::SmallVector<int64_t, 3> coefficients;
        llvm::SmallVector<Value, 3> baseRefs;
        llvm::SmallVector<int64_t, 3> offsets;
        llvm::SmallVector<std::shared_ptr<ElementwiseIndexExpr>, 3>
            indexExpressions;
        for (unsigned arrayDim = 0; arrayDim < indices.size(); ++arrayDim) {
          std::optional<unsigned> matchedDimension;
          std::optional<AffineIndexMatch> matchedIndex;

          // A full-rank access retains the ordinary positional mapping. A
          // lower-rank read may project any one logical loop dimension.
          if (indices.size() == expectedRank) {
            matchedDimension = arrayDim;
            matchedIndex = matchAffineIndex(indices[arrayDim],
                                            expectedIndexMemrefs[arrayDim]);
          } else {
            for (unsigned kernelDim = 0; kernelDim < expectedRank;
                 ++kernelDim) {
              if (auto index = matchAffineIndex(
                      indices[arrayDim], expectedIndexMemrefs[kernelDim])) {
                if (matchedDimension) {
                  reason = "array subscript ambiguously matches two loop "
                           "dimensions";
                  return false;
                }
                matchedDimension = kernelDim;
                matchedIndex = *index;
              }
            }
          }

          if ((!matchedDimension || !matchedIndex) && allowAffineSubscripts) {
            if (auto substituted =
                    matchSubstitutedInduction(indices[arrayDim])) {
              matchedDimension = substituted->first;
              matchedIndex = substituted->second;
            }
          }

          bool acceptedSimple =
              matchedDimension && matchedIndex &&
              (allowAffineSubscripts ||
               (matchedIndex->coefficient == 1 && !matchedIndex->baseRef &&
                matchedIndex->offset == 0));
          if (acceptedSimple) {
            dimensions.push_back(*matchedDimension);
            coefficients.push_back(matchedIndex->coefficient);
            baseRefs.push_back(matchedIndex->baseRef);
            offsets.push_back(matchedIndex->offset);
            indexExpressions.push_back({});
            continue;
          }

          auto expression =
              allowAffineSubscripts && expectedRank == 2
                  ? matchGeneralIndexExpression(
                        indices[arrayDim], expectedIndexMemrefs,
                        arrayCoor.getOperation(), info.indexExpressionOps)
                  : nullptr;
          if (!expression) {
            reason = "array index is not an affine induction-variable "
                     "subscript or supported pack expression";
            return false;
          }

          unsigned loopMask = getIndexExpressionLoopMask(expression);
          unsigned dimension = 0;
          if (loopMask != 0)
            while ((loopMask & (1u << dimension)) == 0)
              ++dimension;
          dimensions.push_back(dimension);
          coefficients.push_back(1);
          baseRefs.push_back({});
          offsets.push_back(0);
          indexExpressions.push_back(std::move(expression));
        }

        Value base = arrayCoor.getMemref();
        for (Operation *user : arrayCoor.getResult().getUsers()) {
          if (auto load = dyn_cast<fir::LoadOp>(user)) {
            info.readArrays.push_back(base);
            info.readValues.push_back(load.getResult());
            info.readDimensions.push_back(dimensions);
            info.readCoefficients.push_back(coefficients);
            info.readBaseRefs.push_back(baseRefs);
            info.readOffsets.push_back(offsets);
            info.readIndexExpressions.push_back(indexExpressions);
          } else if (auto store = dyn_cast<fir::StoreOp>(user)) {
            if (indices.size() != expectedRank) {
              unsigned loopMask = 0;
              for (auto [dimension, expression] :
                   llvm::zip_equal(dimensions, indexExpressions)) {
                loopMask |= expression ? getIndexExpressionLoopMask(expression)
                                       : 1u << dimension;
              }
              unsigned requiredMask = (1u << expectedRank) - 1u;
              if (!allowAffineSubscripts || loopMask != requiredMask) {
                reason = "lower-rank array write subscript does not depend on "
                         "every logical loop dimension";
                return false;
              }
            }
            if (info.writeArray && !allowMultipleWrites) {
              reason = "kernel has more than one write array";
              return false;
            }

            if (!info.writeArray) {
              info.writeArray = base;
              info.storedValue = store.getValue();
            }
            info.writeArrays.push_back(base);
            info.storedValues.push_back(store.getValue());
            info.writeStores.push_back(store.getOperation());
            info.writeDimensions.push_back(dimensions);
            info.writeCoefficients.push_back(coefficients);
            info.writeBaseRefs.push_back(baseRefs);
            info.writeOffsets.push_back(offsets);
            info.writeIndexExpressions.push_back(indexExpressions);
            info.writeConditions.emplace_back(conditions.begin(),
                                              conditions.end());
          } else {
            reason = "array_coor result has unsupported user";
            return false;
          }
        }
        continue;
      }

      if (auto loop = dyn_cast<fir::DoLoopOp>(operation)) {
        if (!allowAffineSubscripts || expectedRank != 2 ||
            !isConstantIntegerValue(loop.getStep(), 1)) {
          reason = "nested stencil loop is not a supported fixed expansion";
          return false;
        }

        auto lower = matchLogicalCoordinate(loop.getLowerBound());
        auto upper = matchLogicalCoordinate(loop.getUpperBound());
        if (!lower || !upper || lower->first != upper->first ||
            lower->second.coefficient != 1 || upper->second.coefficient != 1 ||
            lower->second.baseRef || upper->second.baseRef ||
            upper->second.offset != lower->second.offset + 1) {
          reason = "nested stencil loop must span coordinate:coordinate+1";
          return false;
        }

        Value inductionMemref = findInductionMemref(loop);
        if (!inductionMemref) {
          reason = "nested stencil loop has no induction-variable storage";
          return false;
        }

        recordStructuralOp(loop.getOperation());
        if (Block *loopBody = loop.getBody()) {
          for (Operation &nestedOperation : loopBody->getOperations()) {
            auto store = dyn_cast<fir::StoreOp>(nestedOperation);
            if (store &&
                sameValueAfterFirConvert(store.getMemref(), inductionMemref)) {
              recordStructuralOp(store.getOperation());
              break;
            }
          }
        }

        bool afterLoop = false;
        for (Operation &sibling : current->getOperations()) {
          if (&sibling == loop.getOperation()) {
            afterLoop = true;
            continue;
          }
          if (!afterLoop)
            continue;
          auto store = dyn_cast<fir::StoreOp>(sibling);
          if (store &&
              sameValueAfterFirConvert(store.getMemref(), inductionMemref)) {
            recordStructuralOp(store.getOperation());
            break;
          }
        }

        for (int64_t iteration = 0; iteration != 2; ++iteration) {
          InductionSubstitutions nestedSubstitutions = substitutions;
          nestedSubstitutions.push_back({inductionMemref, lower->first,
                                         lower->second.offset + iteration});
          if (!collectBlock(loop.getBody(), conditions,
                            std::move(nestedSubstitutions)))
            return false;
        }
        continue;
      }

      auto ifOp = dyn_cast<fir::IfOp>(operation);
      if (!ifOp)
        continue;

      WriteConditions thenConditions = conditions;
      thenConditions.push_back(
          {ifOp.getOperation(), ifOp.getCondition(), true});
      for (Block &nested : ifOp.getThenRegion())
        if (!collectBlock(&nested, thenConditions, substitutions))
          return false;

      WriteConditions elseConditions = conditions;
      elseConditions.push_back(
          {ifOp.getOperation(), ifOp.getCondition(), false});
      for (Block &nested : ifOp.getElseRegion())
        if (!collectBlock(&nested, elseConditions, substitutions))
          return false;
    }
    return true;
  };

  if (!collectBlock(body, {}, {}))
    return false;

  if (requireWriteArray && !info.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  llvm::SmallVector<Value> uniqueReadArrays;
  for (Value array : info.readArrays)
    getOrAddArrayIndex(uniqueReadArrays, array);

  if (uniqueReadArrays.size() < minReads ||
      (maxReads != 0 && uniqueReadArrays.size() > maxReads) ||
      info.readArrays.size() != info.readValues.size() ||
      info.readDimensions.size() != info.readValues.size()) {
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
      scalarRef = getReadOnlyScalarCaptureRef(mulRhs);
    } else {
      scaledArrayIndex = findReadValueIndex(info.readValues, mulRhs);
      if (scaledArrayIndex >= 0)
        scalarRef = getReadOnlyScalarCaptureRef(mulLhs);
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

  if (!collectArrayAccessesFromBody(body, 1, indexMemrefs, info, reason, 0, 0,
                                    true, true))
    return false;

  if (!info.writeArray) {
    reason = "kernel has no write array";
    return false;
  }

  bool canonicalPointwise = isConstantIntegerValue(loop.getLowerBound(), 1) &&
                            !info.hasNonIdentitySubscript() &&
                            !info.readArrays.empty() &&
                            info.storedValues.size() == 1;
  if (!canonicalPointwise)
    return detectMultiExpr1D(k, info, reason);

  // The legacy pointwise ABI and its specialized emitters use one element
  // type for every pointer. Route heterogeneous arrays through MultiExpr1D,
  // whose array arguments and accesses retain their individual element types.
  ElementType outputType = getArrayElementType(info.writeArray);
  bool homogeneousArrays =
      llvm::all_of(info.readArrays,
                   [&](Value array) {
                     return getArrayElementType(array) == outputType;
                   }) &&
      llvm::all_of(info.writeArrays, [&](Value array) {
        return getArrayElementType(array) == outputType;
      });
  if (!homogeneousArrays)
    return detectMultiExpr1D(k, info, reason);

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

  // Generic 1-D expression tree. Array and scalar captures are dynamically
  // sized by the v2 launch ABI.
  //
  //   c(i) = alpha * a(i) + beta * b(i)
  //   c(i) = (a(i) + b(i)) * alpha
  //
  // One-read expressions such as c(i) = a(i) + 1.0 are supported.
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
                                    reason, 0, 0, true, true))
    return false;

  bool canonicalPointwise =
      isConstantIntegerValue(innerLoop.getLowerBound(), 1) &&
      isConstantIntegerValue(k.outerLoop.getLowerBound(), 1) &&
      !info.hasNonIdentitySubscript() && info.storedValues.size() == 1;

  if (!canonicalPointwise)
    return detectStencil2D(k, info, reason);

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

  std::string stencilReason;
  if (detectStencil2D(k, info, stencilReason))
    return true;

  reason = "unsupported 2-D expression; binary failure: ";
  reason += binaryReason;
  reason += "; expression-tree failure: ";
  reason += exprReason;
  reason += "; stencil failure: ";
  reason += stencilReason;

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

static std::optional<llvm::SmallVector<Value>>
getReductionScalarsFromLaunch(fir::fnacc::LaunchOp launchOp) {
  auto slotsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_slots");
  if (!slotsAttr || slotsAttr.asArrayRef().empty())
    return std::nullopt;

  auto packVars = launchOp.getPackVars();
  llvm::SmallVector<Value> results;
  for (int32_t slot : slotsAttr.asArrayRef()) {
    if (slot < 0 || static_cast<unsigned>(slot) >= packVars.size())
      return std::nullopt;
    results.push_back(packVars[slot]);
  }
  return results;
}

static std::optional<llvm::SmallVector<ReductionOperator>>
getReductionOperatorsFromLaunch(fir::fnacc::LaunchOp launchOp,
                                unsigned resultCount) {
  auto opsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_ops");
  llvm::SmallVector<ReductionOperator> results;
  if (!opsAttr || opsAttr.asArrayRef().empty()) {
    results.assign(resultCount, ReductionOperator::Add);
    return results;
  }
  if (opsAttr.asArrayRef().size() != resultCount)
    return std::nullopt;

  for (int32_t encoded : opsAttr.asArrayRef()) {
    switch (encoded) {
    case 0:
      results.push_back(ReductionOperator::Add);
      break;
    case 1:
      results.push_back(ReductionOperator::Multiply);
      break;
    case 2:
      results.push_back(ReductionOperator::Min);
      break;
    case 3:
      results.push_back(ReductionOperator::Max);
      break;
    default:
      return std::nullopt;
    }
  }
  return results;
}

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

  for (Value scalar : k.scalarRefs) {
    if (getScalarRefElementType(scalar) != type) {
      reason = "all reduction scalar captures must match the element type";
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

static std::optional<Value> getReductionTerm(Value value,
                                             Value reductionScalarRef,
                                             ReductionOperator reductionOp,
                                             std::string &reason) {
  value = stripFirConvert(value);
  Operation *op = value.getDefiningOp();
  if (!reductionOperationMatches(op, reductionOp)) {
    reason = "reduction store value does not match directive operator";
    return std::nullopt;
  }

  unsigned lhsIndex = op->getName().getStringRef() == "arith.select" ? 1 : 0;
  unsigned rhsIndex = lhsIndex + 1;
  if (op->getNumOperands() <= rhsIndex) {
    reason = "reduction operation has too few operands";
    return std::nullopt;
  }

  Value lhs = stripFirConvert(op->getOperand(lhsIndex));
  Value rhs = stripFirConvert(op->getOperand(rhsIndex));
  if (valueIsLoadOfMemref(lhs, reductionScalarRef))
    return rhs;
  if (valueIsLoadOfMemref(rhs, reductionScalarRef))
    return lhs;

  reason = "reduction operation does not include previous scalar value";
  return std::nullopt;
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
recognizeMultiReduction2D(fir::fnacc::LaunchOp launchOp) {
  std::optional<llvm::SmallVector<Value>> reductionScalars =
      getReductionScalarsFromLaunch(launchOp);
  if (!reductionScalars || reductionScalars->empty())
    return fail(launchOp,
                "2-D reduction recognition requires at least one result");

  std::optional<llvm::SmallVector<ReductionOperator>> reductionOperators =
      getReductionOperatorsFromLaunch(launchOp, reductionScalars->size());
  if (!reductionOperators)
    return fail(launchOp, "multi-reduction metadata is inconsistent");
  if (!llvm::all_of(*reductionOperators, [&](ReductionOperator op) {
        return op == reductionOperators->front();
      }))
    return fail(launchOp, "fused multi-reduction currently requires one common "
                          "operator");

  Region &region = launchOp.getRegion();
  if (region.empty())
    return fail(launchOp, "multi-reduction launch region is empty");

  fir::DoLoopOp outerLoop;
  for (Operation &operation : region.front()) {
    auto loop = dyn_cast<fir::DoLoopOp>(operation);
    if (!loop)
      continue;
    if (outerLoop)
      return fail(&operation, "multi-reduction expected one top-level loop");
    outerLoop = loop;
  }
  if (!outerLoop)
    return fail(launchOp, "multi-reduction found no top-level loop");

  fir::DoLoopOp innerLoop;
  for (Operation &operation : outerLoop.getBody()->getOperations()) {
    auto loop = dyn_cast<fir::DoLoopOp>(operation);
    if (!loop)
      continue;
    if (innerLoop)
      return fail(&operation,
                  "multi-reduction expected one logical inner loop");
    innerLoop = loop;
  }
  if (!innerLoop)
    return fail(outerLoop.getOperation(),
                "multi-reduction found no logical inner loop");

  std::string reason;
  if (!verifyLoopLowerBoundAndStep(outerLoop, "multi-reduction outer", reason))
    return fail(outerLoop.getOperation(), reason);
  if (!verifyLoopLowerBoundAndStep(innerLoop, "multi-reduction inner", reason))
    return fail(innerLoop.getOperation(), reason);

  Value innerIndMemref = findInductionMemref(innerLoop);
  Value outerIndMemref = findInductionMemref(outerLoop);
  if (!innerIndMemref || !outerIndMemref)
    return fail(innerLoop.getOperation(),
                "multi-reduction induction storage could not be determined");

  ElementwiseExtentSource extentX = getLoopExtentSource(innerLoop);
  ElementwiseExtentSource extentY = getLoopExtentSource(outerLoop);
  if (extentX.kind == ElementwiseExtentSourceKind::Unknown ||
      extentY.kind == ElementwiseExtentSourceKind::Unknown)
    return fail(innerLoop.getOperation(),
                "multi-reduction extents could not be determined");

  ArrayAccessInfo accesses;
  llvm::SmallVector<Value, 2> inductionMemrefs{innerIndMemref, outerIndMemref};
  if (!collectArrayAccessesFromBody(
          innerLoop.getBody(), 2, inductionMemrefs, accesses, reason,
          /*minReads=*/1, /*maxReads=*/0, /*allowMultipleWrites=*/true,
          /*allowAffineSubscripts=*/true, /*requireWriteArray=*/false))
    return fail(innerLoop.getOperation(), reason);
  for (const auto &expressions : accesses.readIndexExpressions)
    if (llvm::any_of(expressions,
                     [](const auto &expression) { return bool(expression); }))
      return fail(innerLoop.getOperation(),
                  "multi-reduction general pack index expressions are not "
                  "supported");

  ElementType elementType = getScalarRefElementType(reductionScalars->front());
  if (elementType == ElementType::Unknown)
    return fail(innerLoop.getOperation(),
                "multi-reduction result has unsupported element type");
  for (Value scalar : *reductionScalars)
    if (getScalarRefElementType(scalar) != elementType)
      return fail(innerLoop.getOperation(),
                  "all multi-reduction results must have the same type");

  ElementwiseKernel kernel;
  kernel.kind = ElementwiseKernelKind::MultiReduction2D;
  kernel.rank = 2;
  kernel.outerLoop = outerLoop;
  kernel.innerLoop = innerLoop;
  kernel.extentX = extentX;
  kernel.extentY = extentY;
  kernel.loopLowerX = getLoopLowerSource(innerLoop);
  kernel.loopLowerY = getLoopLowerSource(outerLoop);
  kernel.innerIndMemref = innerIndMemref;
  kernel.outerIndMemref = outerIndMemref;
  kernel.elementType = elementType;
  kernel.reductionOperator = reductionOperators->front();
  kernel.reductionScalarRef = reductionScalars->front();

  for (Operation *operation : accesses.indexExpressionOps)
    markConsumed(kernel, operation);

  for (unsigned index = 0; index < accesses.readValues.size(); ++index) {
    Value array = accesses.readArrays[index];
    ElementType arrayType = getArrayElementType(array);
    if (arrayType != elementType)
      return fail(innerLoop.getOperation(),
                  "multi-reduction input arrays must match the result type");
    getOrAddArrayIndex(kernel.readArrays, array);
    unsigned argumentIndex = getOrAddArrayArgument(kernel, array, true, false);

    ElementwiseArrayAccess access;
    access.array = array;
    access.loadedValue = accesses.readValues[index];
    access.arrayArgumentIndex = argumentIndex;
    access.elementType = arrayType;
    access.dimensions = accesses.readDimensions[index];
    access.coefficients = accesses.readCoefficients[index];
    access.baseIndices =
        getAffineBaseIndices(kernel, accesses.readBaseRefs[index]);
    access.offsets = accesses.readOffsets[index];
    access.indexExpressions = accesses.readIndexExpressions[index];
    bindIndexExpressionCaptures(kernel, access.indexExpressions);
    kernel.arrayAccesses.push_back(std::move(access));
  }

  llvm::SmallVector<fir::StoreOp> reductionStores(reductionScalars->size());
  for (Operation &operation : innerLoop.getBody()->getOperations()) {
    auto store = dyn_cast<fir::StoreOp>(operation);
    if (!store)
      continue;
    for (auto [index, scalar] : llvm::enumerate(*reductionScalars)) {
      if (!sameValueAfterFirConvert(store.getMemref(), scalar))
        continue;
      if (reductionStores[index])
        return fail(store.getOperation(),
                    "multi-reduction result is stored more than once");
      reductionStores[index] = store;
    }
  }

  for (auto [index, scalar] : llvm::enumerate(*reductionScalars)) {
    fir::StoreOp store = reductionStores[index];
    if (!store)
      return fail(innerLoop.getOperation(),
                  "multi-reduction result has no update store");

    std::optional<Value> term = getReductionTerm(
        store.getValue(), scalar, (*reductionOperators)[index], reason);
    if (!term)
      return fail(store.getOperation(), reason);
    if (index == 0)
      kernel.computeOp = stripFirConvert(*term).getDefiningOp();

    std::unique_ptr<ElementwiseExpr> expression =
        recognizeElementwiseExpr(*term, accesses, kernel, reason);
    if (!expression)
      return fail(store.getOperation(), reason);
    if (expression->resultKind != ElementwiseExprResultKind::Element ||
        expression->elementType != elementType)
      return fail(store.getOperation(),
                  "multi-reduction term has an incompatible type");

    ElementwiseReductionOutput output;
    output.scalarRef = scalar;
    output.reductionOperator = (*reductionOperators)[index];
    output.expression = std::move(expression);
    kernel.reductionOutputs.push_back(std::move(output));
    markConsumed(kernel, store.getOperation());
    markLaunchLocalBackwardSlice(kernel, launchOp, store.getValue());
  }

  for (Value scalar : kernel.scalarRefs)
    if (getScalarRefElementType(scalar) != elementType)
      return fail(innerLoop.getOperation(),
                  "multi-reduction scalar captures must match result type");

  markLoopBounds(kernel, launchOp, outerLoop);
  markLoopBounds(kernel, launchOp, innerLoop);
  markInductionStore(kernel, outerLoop, outerIndMemref);
  markInductionStore(kernel, innerLoop, innerIndMemref);
  markPostLoopInductionUpdate(kernel, launchOp, outerLoop, outerIndMemref);
  markPostLoopInductionUpdate(kernel, launchOp, innerLoop, innerIndMemref);
  return ElementwiseRecognitionResult::success(std::move(kernel));
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
  k.loopLowerX = getLoopLowerSource(loop);
  k.innerIndMemref = indMemref;
  k.reductionScalarRef = *reductionScalar;
  k.reductionOperator = *reductionOp;
  k.writeArray = {};
  k.scalarRefs.clear();

  switch (extentX.kind) {
  case ElementwiseExtentSourceKind::ConstantInteger:
    break;
  case ElementwiseExtentSourceKind::Value:
    markLaunchLocalBackwardSlice(k, launchOp, extentX.value);
    break;

  case ElementwiseExtentSourceKind::LoadIntegerRef:
  case ElementwiseExtentSourceKind::BoxDim:
  case ElementwiseExtentSourceKind::BoxLowerBound:
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

  std::string expressionReason;
  std::optional<Value> reductionTerm =
      getReductionTerm(reductionStore.getValue(), *reductionScalar,
                       *reductionOp, expressionReason);
  if (reductionTerm) {
    ArrayAccessInfo accesses;
    for (const ReductionArrayLoadInfo &load : loads) {
      accesses.readArrays.push_back(load.arrayBase);
      accesses.readValues.push_back(load.loadedValue);
    }

    k.readArrays.clear();
    k.scalarRefs.clear();
    k.privateScalarRefs.clear();
    k.expression =
        recognizeElementwiseExpr(*reductionTerm, accesses, k, expressionReason);
    if (k.expression) {
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
      k.computeOp = stripFirConvert(*reductionTerm).getDefiningOp();
      if (!getOrCheckReductionElementType(k, expressionReason))
        return fail(loop.getOperation(), expressionReason);
      return ElementwiseRecognitionResult::success(std::move(k));
    }
  }

  reason = "unsupported reduction expression; dot failure: ";
  reason += dotReason;
  reason += "; simple reduction failure: ";
  reason += simpleReason;
  reason += "; expression-tree failure: ";
  reason += expressionReason;

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
  if (!isConstantIntegerValue(jLoop.getLowerBound(), 1) ||
      !isConstantIntegerValue(iLoop.getLowerBound(), 1) ||
      !isConstantIntegerValue(pLoop.getLowerBound(), 1))
    return fail(iLoop.getOperation(),
                "matmul loop lower bounds must be constant 1");

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
  k.loopLowerX = getLoopLowerSource(loop);
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
  k.loopLowerY = getLoopLowerSource(outer);
  k.loopLowerX = getLoopLowerSource(inner);
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

static bool hasDirectNestedLoop(fir::fnacc::LaunchOp launchOp) {
  Region &region = launchOp.getRegion();
  if (region.empty())
    return false;

  for (Operation &operation : region.front()) {
    auto outerLoop = dyn_cast<fir::DoLoopOp>(operation);
    if (!outerLoop)
      continue;
    return llvm::any_of(
        outerLoop.getBody()->getOperations(),
        [](Operation &nested) { return isa<fir::DoLoopOp>(nested); });
  }
  return false;
}

ElementwiseRecognitionResult
recognizeElementwiseKernel(fir::fnacc::LaunchOp launchOp) {
  // Reductions are explicitly marked by lowering with fnacc.reduction_slots.
  // Do not try reduction recognition on ordinary elementwise/matmul launches,
  // otherwise diagnostics become noisy and misleading.
  if (launchOp->hasAttr("fnacc.reduction_slots")) {
    auto slots =
        launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_slots");
    bool hasMultipleResults = slots && slots.asArrayRef().size() > 1;
    if (hasMultipleResults || hasDirectNestedLoop(launchOp)) {
      auto multi = recognizeMultiReduction2D(launchOp);
      if (multi.succeeded())
        return validateRecognizedKernel(launchOp, std::move(multi));
      if (multi.getFailure().reason ==
          "multi-reduction found no logical inner loop")
        return fail(
            launchOp,
            "not a supported FNACC reduction kernel; reduction recognition "
            "requires exactly one reduction scalar");
      std::string reason =
          hasMultipleResults ? "not a supported FNACC multi-reduction kernel; "
                             : "not a supported FNACC 2-D reduction kernel; ";
      reason += multi.getFailure().reason;
      return fail(launchOp, reason);
    }

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
       kernel.kind == ElementwiseKernelKind::MultiExpr1D ||
       kernel.kind == ElementwiseKernelKind::Expr2D);

  if (isF64ScalarElementwise || isReductionKernelKind(kernel.kind) ||
      kernel.kind == ElementwiseKernelKind::MatMul2D)
    return requestedParallelSubgroups;

  return 1;
}

static llvm::SmallVector<unsigned>
getKernelParameterSlotsForValue(const ElementwiseKernel &kernel, Value value) {
  llvm::SmallVector<unsigned> slots;

  if (usesVariadicLaunchABI(kernel.kind)) {
    unsigned scalarBaseSlot = 0;
    if (!kernel.outputs.empty() || !kernel.reductionOutputs.empty()) {
      for (unsigned i = 0; i < kernel.arrayArguments.size(); ++i)
        if (sameArrayBase(kernel.arrayArguments[i].array, value))
          slots.push_back(i);
      scalarBaseSlot = kernel.arrayArguments.size() +
                       (kernel.reductionOutputs.empty() ? 0u : 1u);
    } else {
      for (unsigned i = 0; i < kernel.readArrays.size(); ++i)
        if (sameArrayBase(kernel.readArrays[i], value))
          slots.push_back(i);
      if (kernel.writeArray && sameArrayBase(kernel.writeArray, value))
        slots.push_back(kernel.readArrays.size());
      scalarBaseSlot = kernel.readArrays.size() + (kernel.writeArray ? 1u : 0u);
    }

    for (unsigned i = 0; i < kernel.scalarRefs.size(); ++i)
      if (sameValueAfterFirConvert(kernel.scalarRefs[i], value))
        slots.push_back(scalarBaseSlot + i);
    for (unsigned i = 0; i < kernel.indexRefs.size(); ++i)
      if (sameValueAfterFirConvert(kernel.indexRefs[i], value))
        slots.push_back(scalarBaseSlot + kernel.scalarRefs.size() + i);
    return slots;
  }

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
                               ElementType elementType, llvm::StringRef name,
                               int32_t arrayIndex = -1, int32_t dimension = -1,
                               int32_t scalarIndex = -1) {
  FNACCKernelParameter parameter;
  parameter.slot = abi.parameters.size();
  parameter.role = role;
  parameter.passing = passing;
  parameter.elementType = elementType;
  parameter.name = name.str();
  parameter.arrayIndex = arrayIndex;
  parameter.scalarIndex = scalarIndex;
  parameter.dimension = dimension;
  abi.parameters.push_back(std::move(parameter));
}

static FNACCKernelABI buildKernelABI(fir::fnacc::LaunchOp launchOp,
                                     const ElementwiseKernel &kernel) {
  FNACCKernelABI abi;
  bool isReduction = isReductionKernelKind(kernel.kind);

  if (usesVariadicLaunchABI(kernel.kind)) {
    auto findArrayIndex = [&](Value value) -> int32_t {
      for (auto [index, argument] : llvm::enumerate(kernel.arrayArguments))
        if (sameArrayBase(argument.array, value))
          return static_cast<int32_t>(index);
      llvm_unreachable("variadic kernel value has no array binding");
    };

    if (!kernel.outputs.empty() || !kernel.reductionOutputs.empty()) {
      for (auto [index, array] : llvm::enumerate(kernel.arrayArguments)) {
        FNACCKernelParameterRole role = FNACCKernelParameterRole::Read;
        if (array.read && array.write)
          role = FNACCKernelParameterRole::ReadWrite;
        else if (array.write)
          role = FNACCKernelParameterRole::Write;
        appendABIParameter(
            abi, role, FNACCKernelParameterPassing::DevicePointer,
            array.elementType, "array" + std::to_string(index), index);
      }
      if (!kernel.reductionOutputs.empty())
        appendABIParameter(abi, FNACCKernelParameterRole::Partials,
                           FNACCKernelParameterPassing::DevicePointer,
                           kernel.elementType, "partials");
    } else {
      for (unsigned i = 0; i < kernel.readArrays.size(); ++i)
        appendABIParameter(abi, FNACCKernelParameterRole::Read,
                           FNACCKernelParameterPassing::DevicePointer,
                           kernel.elementType, "read" + std::to_string(i),
                           findArrayIndex(kernel.readArrays[i]));
      if (isReduction) {
        appendABIParameter(abi, FNACCKernelParameterRole::Partials,
                           FNACCKernelParameterPassing::DevicePointer,
                           kernel.elementType, "partials");
      } else {
        appendABIParameter(abi, FNACCKernelParameterRole::Write,
                           FNACCKernelParameterPassing::DevicePointer,
                           kernel.elementType, "write",
                           findArrayIndex(kernel.writeArray));
      }
    }

    for (unsigned i = 0; i < kernel.scalarRefs.size(); ++i)
      appendABIParameter(abi, FNACCKernelParameterRole::Scalar,
                         FNACCKernelParameterPassing::Value, kernel.elementType,
                         "scalar" + std::to_string(i), -1, -1, i);

    for (unsigned i = 0; i < kernel.indexRefs.size(); ++i) {
      unsigned scalarIndex = kernel.scalarRefs.size() + i;
      appendABIParameter(abi, FNACCKernelParameterRole::Scalar,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "index" + std::to_string(i), -1, -1, scalarIndex);
    }

    appendABIParameter(abi, FNACCKernelParameterRole::ExtentX,
                       FNACCKernelParameterPassing::Value, ElementType::I32,
                       "extent_x");
    if (kernel.rank >= 2)
      appendABIParameter(abi, FNACCKernelParameterRole::ExtentY,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "extent_y");
    if (kernel.kind == ElementwiseKernelKind::MatMul2D)
      appendABIParameter(abi, FNACCKernelParameterRole::ExtentZ,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "extent_k");

    if (kernel.kind == ElementwiseKernelKind::MultiExpr1D ||
        (isReduction && kernel.rank == 1)) {
      appendABIParameter(abi, FNACCKernelParameterRole::LoopLowerX,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "loop_lower_x");

      for (unsigned array = 0; array < kernel.arrayArguments.size(); ++array) {
        appendABIParameter(abi, FNACCKernelParameterRole::ArrayLowerBound,
                           FNACCKernelParameterPassing::Value, ElementType::I32,
                           "array" + std::to_string(array) + "_lower0", array,
                           0);
        appendABIParameter(abi, FNACCKernelParameterRole::ArrayStride,
                           FNACCKernelParameterPassing::Value, ElementType::I32,
                           "array" + std::to_string(array) + "_stride0", array,
                           0);
      }
    } else if (kernel.kind == ElementwiseKernelKind::Stencil2D ||
               kernel.kind == ElementwiseKernelKind::MultiReduction2D) {
      appendABIParameter(abi, FNACCKernelParameterRole::LoopLowerX,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "loop_lower_x");
      appendABIParameter(abi, FNACCKernelParameterRole::LoopLowerY,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "loop_lower_y");

      for (unsigned array = 0; array < kernel.arrayArguments.size(); ++array) {
        for (unsigned dim = 0; dim < 2; ++dim)
          appendABIParameter(
              abi, FNACCKernelParameterRole::ArrayLowerBound,
              FNACCKernelParameterPassing::Value, ElementType::I32,
              "array" + std::to_string(array) + "_lower" + std::to_string(dim),
              array, dim);
        for (unsigned dim = 0; dim < 2; ++dim)
          appendABIParameter(
              abi, FNACCKernelParameterRole::ArrayStride,
              FNACCKernelParameterPassing::Value, ElementType::I32,
              "array" + std::to_string(array) + "_stride" + std::to_string(dim),
              array, dim);
      }
    }
  } else {
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
                           FNACCKernelParameterPassing::Value,
                           kernel.elementType, "scalar" + std::to_string(i));

      appendABIParameter(abi, FNACCKernelParameterRole::ExtentX,
                         FNACCKernelParameterPassing::Value, ElementType::I32,
                         "extent_x");

      if (kernel.rank == 2) {
        appendABIParameter(abi, FNACCKernelParameterRole::ExtentY,
                           FNACCKernelParameterPassing::Value, ElementType::I32,
                           "extent_y");

        if (kernel.kind == ElementwiseKernelKind::MatMul2D)
          appendABIParameter(abi, FNACCKernelParameterRole::ExtentZ,
                             FNACCKernelParameterPassing::Value,
                             ElementType::I32, "extent_k");
      }
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
         kind == ElementwiseKernelKind::ReductionMax1D ||
         kind == ElementwiseKernelKind::MultiReduction2D;
}

bool usesVariadicLaunchABI(ElementwiseKernelKind kind) {
  switch (kind) {
  case ElementwiseKernelKind::BinaryArrayArray:
  case ElementwiseKernelKind::Saxpy1D:
  case ElementwiseKernelKind::Expr1D:
  case ElementwiseKernelKind::MultiExpr1D:
  case ElementwiseKernelKind::Expr2D:
  case ElementwiseKernelKind::Stencil2D:
  case ElementwiseKernelKind::MatMul2D:
  case ElementwiseKernelKind::ReductionSum1D:
  case ElementwiseKernelKind::ReductionDot1D:
  case ElementwiseKernelKind::ReductionProduct1D:
  case ElementwiseKernelKind::ReductionMin1D:
  case ElementwiseKernelKind::ReductionMax1D:
  case ElementwiseKernelKind::MultiReduction2D:
    return true;
  }
  llvm_unreachable("unknown FNACC kernel kind");
}

llvm::StringRef fnaccKernelKindName(ElementwiseKernelKind kind) {
  switch (kind) {
  case ElementwiseKernelKind::BinaryArrayArray:
    return "binary";
  case ElementwiseKernelKind::Saxpy1D:
    return "saxpy1d";
  case ElementwiseKernelKind::Expr1D:
    return "expr1d";
  case ElementwiseKernelKind::MultiExpr1D:
    return "multi_expr1d";
  case ElementwiseKernelKind::Expr2D:
    return "expr2d";
  case ElementwiseKernelKind::Stencil2D:
    return "stencil2d";
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
  case ElementwiseKernelKind::MultiReduction2D:
    return "reduction_multi2d";
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
  plan.usesVariadicABI = usesVariadicLaunchABI(plan.kernel.kind);
  plan.copyBackWrites = !launchOp->hasAttr("fnacc.no_copyback");
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

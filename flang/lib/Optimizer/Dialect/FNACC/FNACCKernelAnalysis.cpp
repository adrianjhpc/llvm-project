#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"

#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APFloat.h"

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

    /// Return the i32 reference from a loop upper bound.
    ///
    /// Supports both shapes:
    ///
    ///   %ub = fir.load %n : !fir.ref<i32>
    ///   fir.do_loop ... to %ub ...
    ///
    /// and:
    ///
    ///   %load = fir.load %n : !fir.ref<i32>
    ///   %ub = fir.convert %load : i32 -> index
    ///   fir.do_loop ... to %ub ...
    static std::optional<Value> getI32UpperBoundRef(fir::DoLoopOp loop) {
      Value ub = loop.getUpperBound();

      auto getI32RefFromLoad = [](Value v) -> std::optional<Value> {
	if (auto load = v.getDefiningOp<fir::LoadOp>()) {
	  Value memref = load.getMemref();

	  if (auto refTy = dyn_cast<fir::ReferenceType>(memref.getType())) {
	    if (refTy.getEleTy().isInteger(32))
	      return memref;
	  }
	}

	return std::nullopt;
      };

      if (auto direct = getI32RefFromLoad(ub))
	return direct;

      while (auto cvt = ub.getDefiningOp<fir::ConvertOp>())
	ub = cvt.getValue();

      if (auto converted = getI32RefFromLoad(ub))
	return converted;

      return std::nullopt;
    }

    /// Strip FIR converts from a scalar value.
    static Value stripFirConvert(Value v) {
      while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
	v = cvt.getValue();

      return v;
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
    static bool verifyLoopLowerBoundAndStep(fir::DoLoopOp loop,
					    StringRef loopName,
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

    static bool isF32ArrayReference(Value v) {
      auto refTy = dyn_cast<fir::ReferenceType>(v.getType());
      if (!refTy)
	return false;

      auto arrTy = dyn_cast<fir::SequenceType>(refTy.getEleTy());
      if (!arrTy)
	return false;

      return arrTy.getEleTy().isF32();
    }

    static bool allArraysAreF32(const ElementwiseKernel &k) {
      if (!k.writeArray || !isF32ArrayReference(k.writeArray))
	return false;

      for (Value v : k.readArrays) {
	if (!isF32ArrayReference(v))
	  return false;
      }

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

    static std::optional<Value> getScalarF32RefFromValue(Value v) {
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

      if (!refTy.getEleTy().isF32())
	return std::nullopt;

      return memref;
    }

    static std::unique_ptr<ElementwiseExpr>
    makeExpr(ElementwiseExprKind kind) {
      auto expr = std::make_unique<ElementwiseExpr>();
      expr->kind = kind;
      return expr;
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

    static std::optional<double> getF32ConstantValue(Value v) {
      v = stripFirConvert(v);

      auto constant = v.getDefiningOp<arith::ConstantOp>();
      if (!constant)
	return std::nullopt;

      auto floatAttr = dyn_cast<FloatAttr>(constant.getValue());
      if (!floatAttr)
	return std::nullopt;

      if (!floatAttr.getType().isF32())
	return std::nullopt;

      return floatAttr.getValueAsDouble();
    }

    static std::unique_ptr<ElementwiseExpr>
    recognizeExpr1D(Value v,
		    const ArrayAccessInfo &info,
		    ElementwiseKernel &k,
		    std::string &reason) {
      v = stripFirConvert(v);

      // Array element load.
      int readIndex = findReadValueIndex(info.readValues, v);
      if (readIndex >= 0) {
	Value arrayBase = info.readArrays[readIndex];
	getOrAddValueIndex(k.readArrays, arrayBase);

	auto expr = makeExpr(ElementwiseExprKind::ArrayLoad);
	expr->source = arrayBase;
	return expr;
      }

      // Scalar f32 load.
      if (auto scalarRef = getScalarF32RefFromValue(v)) {
	getOrAddValueIndex(k.scalarRefs, *scalarRef);

	auto expr = makeExpr(ElementwiseExprKind::ScalarLoad);
	expr->source = *scalarRef;
	return expr;
      }

      // f32 constant.
      if (auto constantValue = getF32ConstantValue(v)) {
	auto expr = makeExpr(ElementwiseExprKind::ConstantF32);
	expr->f32Value = *constantValue;
	return expr;
      }

      Operation *op = v.getDefiningOp();
      if (!op) {
	reason = "expression value has no defining operation";
	return nullptr;
      }

      ElementwiseExprKind exprKind;

      if (isa<arith::AddFOp>(op)) {
	exprKind = ElementwiseExprKind::AddF;
      } else if (isa<arith::SubFOp>(op)) {
	exprKind = ElementwiseExprKind::SubF;
      } else if (isa<arith::MulFOp>(op)) {
	exprKind = ElementwiseExprKind::MulF;
      } else if (isa<arith::DivFOp>(op)) {
	exprKind = ElementwiseExprKind::DivF;
      } else {
	reason = "unsupported operation in 1-D expression tree";
	return nullptr;
      }

      if (op->getNumOperands() != 2) {
	reason = "expression operation is not binary";
	return nullptr;
      }

      auto lhs = recognizeExpr1D(op->getOperand(0), info, k, reason);
      if (!lhs)
	return nullptr;

      auto rhs = recognizeExpr1D(op->getOperand(1), info, k, reason);
      if (!rhs)
	return nullptr;

      auto expr = makeExpr(exprKind);
      expr->operands.push_back(std::move(lhs));
      expr->operands.push_back(std::move(rhs));

      return expr;
    }

    static bool detectGenericExpr1D(ElementwiseKernel &k,
				    const ArrayAccessInfo &info,
				    std::string &reason) {
      k.kind = ElementwiseKernelKind::Expr1D;
      k.rank = 1;
      k.writeArray = info.writeArray;
      k.readArrays.clear();
      k.scalarRefs.clear();
      k.computeOp = info.storedValue.getDefiningOp();

      auto expr = recognizeExpr1D(info.storedValue, info, k, reason);
      if (!expr)
	return false;

      if (k.readArrays.empty()) {
	reason = "expression tree contains no array loads";
	return false;
      }

      if (k.readArrays.size() > 2) {
	reason = "expression tree currently supports at most two read arrays";
	return false;
      }

      if (k.scalarRefs.size() > 2) {
	reason = "expression tree currently supports at most two scalar f32 values";
	return false;
      }

      k.expression = std::move(expr);

      if (!allArraysAreF32(k)) {
	reason = "only f32 arrays are currently supported";
	return false;
      }

      return true;
    }

    static bool collectArrayAccessesFromBody(Block *body,
					     unsigned expectedRank,
					     ArrayRef<Value> expectedIndexMemrefs,
					     ArrayAccessInfo &info,
					     std::string &reason) {
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

      if (info.readArrays.size() != 2 || info.readValues.size() != 2) {
	reason = "kernel expected exactly two read arrays";
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

    static bool detectSaxpy1D(ElementwiseKernel &k,
			      const ArrayAccessInfo &info,
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

      auto tryMatch = [&](Value maybeMulValue,
			  Value maybeAddArrayValue) -> bool {
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
	  scalarRef = getScalarF32RefFromValue(mulRhs);
	} else {
	  scaledArrayIndex = findReadValueIndex(info.readValues, mulRhs);
	  if (scaledArrayIndex >= 0)
	    scalarRef = getScalarF32RefFromValue(mulLhs);
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


    static bool collectArrayAccesses1D(ElementwiseKernel &k,
				       fir::DoLoopOp loop,
				       Value indMemref,
				       std::string &reason) {
      ArrayAccessInfo info;

      Block *body = loop.getBody();
      llvm::SmallVector<Value> indexMemrefs;
      indexMemrefs.push_back(indMemref);

      if (!collectArrayAccessesFromBody(body, 1, indexMemrefs, info, reason))
	return false;

      if (!info.writeArray) {
	reason = "kernel has no write array";
	return false;
      }

      if (info.readArrays.size() != 2) {
	reason = "kernel expected exactly two read arrays";
	return false;
      }

      // Existing direct form:
      //
      //   c(i) = a(i) op b(i)
      std::string binaryReason;
      if (detectDirectBinaryArrayArray(k, info, binaryReason)) {
	if (!allArraysAreF32(k)) {
	  reason = "only f32 arrays are currently supported";
	  return false;
	}

	return true;
      }

      // Existing restricted SAXPY:
      //
      //   c(i) = alpha * a(i) + b(i)
      std::string saxpyReason;
      if (detectSaxpy1D(k, info, saxpyReason)) {
	if (!allArraysAreF32(k)) {
	  reason = "only f32 arrays are currently supported";
	  return false;
	}

	return true;
      }

      // New generic 1-D expression tree.
      //
      // This supports forms such as:
      //
      //   c(i) = alpha * a(i) + beta * b(i)
      //   c(i) = (a(i) + b(i)) * alpha
      //   c(i) = a(i) + 1.0
      std::string exprReason;
      if (detectGenericExpr1D(k, info, exprReason))
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
				       Value innerIndMemref,
				       Value outerIndMemref,
				       std::string &reason) {
      Value writeStoredValue;

      Block *body = innerLoop.getBody();
      if (!body) {
	reason = "inner loop has no body";
	return false;
      }

      for (Operation &op : body->getOperations()) {
	auto ac = dyn_cast<fir::ArrayCoorOp>(op);
	if (!ac)
	  continue;

	auto indices = ac.getIndices();
	if (indices.size() != 2) {
	  reason = "2-D kernel expected array_coor with exactly two indices";
	  return false;
	}

	// FIR for a(i,j) is array_coor %array %i, %j.
	if (!indexIsLoadOf(indices[0], innerIndMemref)) {
	  reason = "first 2-D array index is not the inner loop induction variable";
	  return false;
	}

	if (!indexIsLoadOf(indices[1], outerIndMemref)) {
	  reason = "second 2-D array index is not the outer loop induction variable";
	  return false;
	}

	Value base = ac.getMemref();

	for (Operation *user : ac.getResult().getUsers()) {
	  if (isa<fir::LoadOp>(user)) {
	    k.readArrays.push_back(base);
	  } else if (auto st = dyn_cast<fir::StoreOp>(user)) {
	    if (k.writeArray) {
	      reason = "kernel has more than one write array";
	      return false;
	    }

	    k.writeArray = base;
	    writeStoredValue = st.getValue();
	  } else {
	    reason = "array_coor result has unsupported user";
	    return false;
	  }
	}
      }

      if (!k.writeArray) {
	reason = "kernel has no write array";
	return false;
      }

      if (k.readArrays.size() != 2) {
	reason = "kernel expected exactly two read arrays";
	return false;
      }

      k.computeOp = writeStoredValue.getDefiningOp();

      if (!isSupportedElementwiseCompute(k.computeOp)) {
	reason = "stored value is not a supported binary floating-point op";
	return false;
      }

      if (!allArraysAreF32(k)) {
	reason = "only f32 arrays are currently supported";
	return false;
      }

      return true;
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

      auto nRef = getI32UpperBoundRef(loop);

      if (!nRef)
	return fail(loop.getOperation(),
		    "1-D loop upper bound is not an i32 reference load");

      Value indMemref = findInductionMemref(loop);
      if (!indMemref)
	return fail(loop.getOperation(),
		    "could not find induction variable storage in 1-D loop");

      ElementwiseKernel k;
      k.rank = 1;
      k.loop1D = loop;
      k.nRef = *nRef;
      k.innerIndMemref = indMemref;

      std::string reason;
      if (!collectArrayAccesses1D(k, loop, indMemref, reason))
	return fail(loop.getOperation(), reason);

      return ElementwiseRecognitionResult::success(std::move(k));
    }

    static ElementwiseRecognitionResult
    recognize2D(fir::fnacc::LaunchOp launchOp) {
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

      auto mRef = getI32UpperBoundRef(outer);
      auto nRef = getI32UpperBoundRef(inner);


      if (!mRef)
	return fail(outer.getOperation(),
		    "outer loop upper bound is not an i32 reference load");

      if (!nRef)
	return fail(inner.getOperation(),
		    "inner loop upper bound is not an i32 reference load");

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
      k.mRef = *mRef;
      k.nRef = *nRef;
      k.outerIndMemref = outerIndMemref;
      k.innerIndMemref = innerIndMemref;

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
    // Try 2-D first because a 2-D launch also has one top-level loop.
    auto r2 = recognize2D(launchOp);
    if (r2.succeeded())
      return r2;

    auto r1 = recognize1D(launchOp);
    if (r1.succeeded())
      return r1;

    std::string reason = "not a supported 1-D or 2-D elementwise kernel; ";
    reason += "2-D failure: ";
    reason += r2.getFailure().reason;
    reason += "; 1-D failure: ";
    reason += r1.getFailure().reason;

    return fail(launchOp, reason);
  }

} // namespace fir::fnacc


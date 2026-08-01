#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <optional>

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPULOWERTORUNTIME
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

  //===----------------------------------------------------------------------===//
  // Analysis result
  //===----------------------------------------------------------------------===//

  struct RuntimeKernel {
    int rank = 1;

    // 1-D case:
    fir::DoLoopOp loop1D;

    // 2-D case:
    //   outerLoop = j loop
    //   innerLoop = i loop
    fir::DoLoopOp outerLoop;
    fir::DoLoopOp innerLoop;

    // Runtime extents:
    //   extentX = n
    //   extentY = m
    Value nRef;
    Value mRef;

    // Induction variable storage:
    //   innerIndMemref = i variable
    //   outerIndMemref = j variable
    Value innerIndMemref;
    Value outerIndMemref;

    // Current supported kernel shape:
    //   c = op(a,b)
    llvm::SmallVector<Value> readArrays;
    Value writeArray;

    Operation *computeOp = nullptr;
  };

  struct BlockShape {
    int32_t rank = 1;
    int32_t x = 1024;
    int32_t y = 1;
    int32_t z = 1;
  };

  static BlockShape getBlockShape(fir::fngpu::LaunchOp launchOp) {
    BlockShape shape;

    llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

    if (tiles.empty()) {
      shape.rank = 1;
      shape.x = 1024;
      shape.y = 1;
      shape.z = 1;
      return shape;
    }

    shape.rank = static_cast<int32_t>(tiles.size());

    if (tiles.size() >= 1)
      shape.x = static_cast<int32_t>(tiles[0]);
    if (tiles.size() >= 2)
      shape.y = static_cast<int32_t>(tiles[1]);
    if (tiles.size() >= 3)
      shape.z = static_cast<int32_t>(tiles[2]);

    return shape;
  }

  //===----------------------------------------------------------------------===//
  // Common FIR pattern helpers
  //===----------------------------------------------------------------------===//

  // Upper bounds in the FIR we see look like:
  //
  //   %load = fir.load %nref : !fir.ref<i32>
  //   %ub   = fir.convert %load : (i32) -> index
  //   fir.do_loop ... to %ub ...
  //
  // We return %nref, not %load, because %load may be inside fngpu.launch and
  // would be invalid after the launch is erased.
  static std::optional<Value> getI32UpperBoundRef(fir::DoLoopOp loop) {
    Value ub = loop.getUpperBound();

    if (auto cvt = ub.getDefiningOp<fir::ConvertOp>()) {
      Value src = cvt.getValue();

      if (auto load = src.getDefiningOp<fir::LoadOp>()) {
	Value memref = load.getMemref();

	if (auto refTy = dyn_cast<fir::ReferenceType>(memref.getType())) {
	  if (refTy.getEleTy().isInteger(32))
	    return memref;
	}
      }
    }

    return std::nullopt;
  }

  // The lowered FIR keeps the Fortran induction variable in an alloca/declaration.
  // At loop entry it emits:
  //
  //   fir.store %iter_arg to %i
  //
  // This finds %i.
  static Value findInductionMemref(fir::DoLoopOp loop) {
    if (loop.getRegionIterArgs().empty())
      return {};

    Value iterArg = loop.getRegionIterArgs()[0];

    Block *body = loop.getBody();
    if (!body)
      return {};

    for (Operation &op : body->getOperations()) {
      if (auto st = dyn_cast<fir::StoreOp>(op)) {
	if (st.getValue() == iterArg)
	  return st.getMemref();
      }
    }

    return {};
  }

  // Trace an array_coor index back through:
  //
  //   fir.convert(fir.load(%i))
  //
  // and check it is a load of the expected induction memref.
  static bool indexIsLoadOf(Value v, Value memref) {
    while (true) {
      if (auto cvt = v.getDefiningOp<fir::ConvertOp>()) {
	v = cvt.getValue();
	continue;
      }

      if (auto load = v.getDefiningOp<fir::LoadOp>())
	return load.getMemref() == memref;

      return false;
    }
  }

  static bool isSupportedBinaryCompute(Operation *op) {
    return op &&
      isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(op);
  }

  //===----------------------------------------------------------------------===//
  // 1-D recogniser
  //===----------------------------------------------------------------------===//

  static std::optional<RuntimeKernel>
  recognise1D(fir::fngpu::LaunchOp launchOp) {
    Region &region = launchOp.getRegion();
    if (region.empty())
      return std::nullopt;

    Block &launchBlock = region.front();

    fir::DoLoopOp loop;
    for (Operation &op : launchBlock) {
      if (auto dl = dyn_cast<fir::DoLoopOp>(op)) {
	if (loop)
	  return std::nullopt;
	loop = dl;
      }
    }

    if (!loop)
      return std::nullopt;

    auto nRef = getI32UpperBoundRef(loop);
    if (!nRef)
      return std::nullopt;

    Value indMemref = findInductionMemref(loop);
    if (!indMemref)
      return std::nullopt;

    RuntimeKernel k;
    k.rank = 1;
    k.loop1D = loop;
    k.nRef = *nRef;
    k.innerIndMemref = indMemref;

    Value writeStoredValue;

    Block *body = loop.getBody();
    if (!body)
      return std::nullopt;

    for (Operation &op : body->getOperations()) {
      auto ac = dyn_cast<fir::ArrayCoorOp>(op);
      if (!ac)
	continue;

      auto indices = ac.getIndices();
      if (indices.size() != 1)
	return std::nullopt;

      if (!indexIsLoadOf(indices[0], indMemref))
	return std::nullopt;

      Value base = ac.getMemref();

      for (Operation *user : ac.getResult().getUsers()) {
	if (isa<fir::LoadOp>(user)) {
	  k.readArrays.push_back(base);
	} else if (auto st = dyn_cast<fir::StoreOp>(user)) {
	  if (k.writeArray)
	    return std::nullopt;

	  k.writeArray = base;
	  writeStoredValue = st.getValue();
	} else {
	  return std::nullopt;
	}
      }
    }

    if (!k.writeArray)
      return std::nullopt;

    if (k.readArrays.size() != 2)
      return std::nullopt;

    k.computeOp = writeStoredValue.getDefiningOp();

    if (!isSupportedBinaryCompute(k.computeOp))
      return std::nullopt;

    return k;
  }

  //===----------------------------------------------------------------------===//
  // 2-D recogniser
  //===----------------------------------------------------------------------===//

  static std::optional<RuntimeKernel>
  recognise2D(fir::fngpu::LaunchOp launchOp) {
    Region &region = launchOp.getRegion();
    if (region.empty())
      return std::nullopt;

    Block &launchBlock = region.front();

    // Launch region should contain exactly one outer loop.
    fir::DoLoopOp outer;
    for (Operation &op : launchBlock) {
      if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
	if (outer)
	  return std::nullopt;
	outer = loop;
      }
    }

    if (!outer)
      return std::nullopt;

    Block *outerBody = outer.getBody();
    if (!outerBody)
      return std::nullopt;

    // Outer body should contain exactly one inner loop.
    fir::DoLoopOp inner;
    for (Operation &op : outerBody->getOperations()) {
      if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
	if (inner)
	  return std::nullopt;
	inner = loop;
      }
    }

    if (!inner)
      return std::nullopt;

    auto mRef = getI32UpperBoundRef(outer);
    auto nRef = getI32UpperBoundRef(inner);

    if (!mRef || !nRef)
      return std::nullopt;

    Value outerIndMemref = findInductionMemref(outer); // j
    Value innerIndMemref = findInductionMemref(inner); // i

    if (!outerIndMemref || !innerIndMemref)
      return std::nullopt;

    RuntimeKernel k;
    k.rank = 2;
    k.outerLoop = outer;
    k.innerLoop = inner;
    k.mRef = *mRef;
    k.nRef = *nRef;
    k.outerIndMemref = outerIndMemref;
    k.innerIndMemref = innerIndMemref;

    Value writeStoredValue;

    Block *innerBody = inner.getBody();
    if (!innerBody)
      return std::nullopt;

    for (Operation &op : innerBody->getOperations()) {
      auto ac = dyn_cast<fir::ArrayCoorOp>(op);
      if (!ac)
	continue;

      auto indices = ac.getIndices();
      if (indices.size() != 2)
	return std::nullopt;

      // FIR for a(i,j) has array_coor %array %i, %j.
      if (!indexIsLoadOf(indices[0], innerIndMemref))
	return std::nullopt;

      if (!indexIsLoadOf(indices[1], outerIndMemref))
	return std::nullopt;

      Value base = ac.getMemref();

      for (Operation *user : ac.getResult().getUsers()) {
	if (isa<fir::LoadOp>(user)) {
	  k.readArrays.push_back(base);
	} else if (auto st = dyn_cast<fir::StoreOp>(user)) {
	  if (k.writeArray)
	    return std::nullopt;

	  k.writeArray = base;
	  writeStoredValue = st.getValue();
	} else {
	  return std::nullopt;
	}
      }
    }

    if (!k.writeArray)
      return std::nullopt;

    if (k.readArrays.size() != 2)
      return std::nullopt;

    k.computeOp = writeStoredValue.getDefiningOp();

    if (!isSupportedBinaryCompute(k.computeOp))
      return std::nullopt;

    return k;
  }

  static std::optional<RuntimeKernel>
  recognise(fir::fngpu::LaunchOp launchOp) {
    if (auto k2 = recognise2D(launchOp))
      return k2;

    return recognise1D(launchOp);
  }

  //===----------------------------------------------------------------------===//
  // Runtime declaration helper
  //===----------------------------------------------------------------------===//

  static func::FuncOp getOrCreateRuntimeDecl(ModuleOp module,
					     OpBuilder &builder,
					     Location loc,
					     StringRef name,
					     TypeRange argTypes) {
    if (auto existing = module.lookupSymbol<func::FuncOp>(name))
      return existing;

    OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPointToStart(module.getBody());

    auto fnType = builder.getFunctionType(argTypes, TypeRange{});
    auto fn = func::FuncOp::create(loc, name, fnType);
    fn.setPrivate();

    builder.insert(fn);
    return fn;
  }

  //===----------------------------------------------------------------------===//
  // Pass
  //===----------------------------------------------------------------------===//

  struct FNGPULowerToRuntimePass
    : public fir::fngpu::impl::FNGPULowerToRuntimeBase<
    FNGPULowerToRuntimePass> {
    void runOnOperation() override {
      ModuleOp module = getOperation();
      OpBuilder builder(module.getContext());

      llvm::SmallVector<fir::fngpu::LaunchOp> launches;
      module.walk([&](fir::fngpu::LaunchOp launchOp) {
	launches.push_back(launchOp);
      });

      int kernelId = 0;

      for (fir::fngpu::LaunchOp launchOp : launches) {
	auto k = recognise(launchOp);

	if (!k) {
	  launchOp.emitWarning(
			       "FNGPU runtime lowering currently supports only 1-D or 2-D "
			       "elementwise kernels of the form c=op(a,b); leaving "
			       "fngpu.launch unchanged");
	  ++kernelId;
	  continue;
	}

	Location loc = launchOp.getLoc();

	BlockShape blockShape = getBlockShape(launchOp);

	builder.setInsertionPoint(launchOp);

	Value kernelIdValue =
          arith::ConstantIntOp::create(builder, loc, kernelId, 32);

	Value rankValue =
          arith::ConstantIntOp::create(builder, loc, k->rank, 32);

	Value blockXValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.x, 32);

	Value blockYValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.y, 32);

	Value blockZValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.z, 32);

	Value extentXValue =
          fir::LoadOp::create(builder, loc, k->nRef);

	Value extentYValue;
	if (k->rank == 2) {
	  extentYValue = fir::LoadOp::create(builder, loc, k->mRef);
	} else {
	  extentYValue =
            arith::ConstantIntOp::create(builder, loc, 1, 32);
	}

	Value extentZValue =
          arith::ConstantIntOp::create(builder, loc, 1, 32);

	StringRef runtimeName = "__fngpu_launch_nd_f32";

	llvm::SmallVector<Type> argTypes;
	argTypes.push_back(kernelIdValue.getType());
	argTypes.push_back(rankValue.getType());
	argTypes.push_back(blockXValue.getType());
	argTypes.push_back(blockYValue.getType());
	argTypes.push_back(blockZValue.getType());
	argTypes.push_back(k->readArrays[0].getType());
	argTypes.push_back(k->readArrays[1].getType());
	argTypes.push_back(k->writeArray.getType());
	argTypes.push_back(extentXValue.getType());
	argTypes.push_back(extentYValue.getType());
	argTypes.push_back(extentZValue.getType());

	func::FuncOp callee =
          getOrCreateRuntimeDecl(module, builder, loc, runtimeName, argTypes);

	llvm::SmallVector<Value> operands;
	operands.push_back(kernelIdValue);
	operands.push_back(rankValue);
	operands.push_back(blockXValue);
	operands.push_back(blockYValue);
	operands.push_back(blockZValue);
	operands.push_back(k->readArrays[0]);
	operands.push_back(k->readArrays[1]);
	operands.push_back(k->writeArray);
	operands.push_back(extentXValue);
	operands.push_back(extentYValue);
	operands.push_back(extentZValue);

	builder.setInsertionPoint(launchOp);

	func::CallOp::create(
			     builder,
			     loc,
			     callee.getSymName(),
			     TypeRange{},
			     operands);

	launchOp.erase();

	++kernelId;
      }
    }
  };

} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPULowerToRuntimePass() {
  return std::make_unique<FNGPULowerToRuntimePass>();
}


#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APFloat.h"

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPULOWERTORUNTIME
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kKernelIdAttrName = "fngpu.kernel_id";

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

static int32_t getKernelId(fir::fngpu::LaunchOp launchOp, int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName))
    return static_cast<int32_t>(attr.getInt());

  launchOp.emitWarning("fngpu.launch has no fngpu.kernel_id attribute; using "
                       "walk-order fallback");

  return fallbackId;
}

static func::FuncOp getOrCreateRuntimeDecl(ModuleOp module, OpBuilder &builder,
                                           Location loc, StringRef name,
                                           TypeRange argTypes) {
  auto fnType = builder.getFunctionType(argTypes, TypeRange{});

  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    if (existing.getFunctionType() != fnType) {
      existing.emitError("existing FNGPU runtime declaration has incompatible "
                         "function type for symbol ")
          << name << ". Existing type is " << existing.getFunctionType()
          << ", requested type is " << fnType;
    }

    return existing;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto fn = func::FuncOp::create(loc, name, fnType);
  fn.setPrivate();

  builder.insert(fn);
  return fn;
}

static Type getI8RefType(OpBuilder &builder) {
  return fir::ReferenceType::get(builder.getI8Type());
}

static Value convertToOpaqueRuntimePtr(OpBuilder &builder, Location loc,
                                       Value value) {
  Type i8RefTy = getI8RefType(builder);
  return fir::ConvertOp::create(builder, loc, i8RefTy, value);
}

static void createVoidRuntimeCall(ModuleOp module, OpBuilder &builder,
                                  Location loc, StringRef runtimeName,
                                  ValueRange operands) {
  llvm::SmallVector<Type> argTypes;
  for (Value operand : operands)
    argTypes.push_back(operand.getType());

  func::FuncOp callee =
      getOrCreateRuntimeDecl(module, builder, loc, runtimeName, argTypes);

  func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                       operands);
}

static void lowerFNGPUDataOpsToRuntime(ModuleOp module, OpBuilder &builder) {
  llvm::SmallVector<fir::fngpu::UpdateHostOp> updateHostOps;
  llvm::SmallVector<fir::fngpu::UpdateDeviceOp> updateDeviceOps;
  llvm::SmallVector<fir::fngpu::ReleaseOp> releaseOps;
  llvm::SmallVector<fir::fngpu::ReleaseAllOp> releaseAllOps;

  module.walk(
      [&](fir::fngpu::UpdateHostOp op) { updateHostOps.push_back(op); });

  module.walk(
      [&](fir::fngpu::UpdateDeviceOp op) { updateDeviceOps.push_back(op); });

  module.walk([&](fir::fngpu::ReleaseOp op) { releaseOps.push_back(op); });

  module.walk(
      [&](fir::fngpu::ReleaseAllOp op) { releaseAllOps.push_back(op); });

  for (fir::fngpu::UpdateHostOp op : updateHostOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    Value ptr = convertToOpaqueRuntimePtr(builder, loc, op.getVar());

    createVoidRuntimeCall(module, builder, loc, "__fngpu_update_host",
                          ValueRange{ptr});

    op.erase();
  }

  for (fir::fngpu::UpdateDeviceOp op : updateDeviceOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    Value ptr = convertToOpaqueRuntimePtr(builder, loc, op.getVar());

    createVoidRuntimeCall(module, builder, loc, "__fngpu_update_device",
                          ValueRange{ptr});

    op.erase();
  }

  for (fir::fngpu::ReleaseOp op : releaseOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    for (Value var : op.getVars()) {
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);

      createVoidRuntimeCall(module, builder, loc, "__fngpu_release",
                            ValueRange{ptr});
    }

    op.erase();
  }

  for (fir::fngpu::ReleaseAllOp op : releaseAllOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    createVoidRuntimeCall(module, builder, loc, "__fngpu_release_all",
                          ValueRange{});

    op.erase();
  }
}

struct FNGPULowerToRuntimePass
    : public fir::fngpu::impl::FNGPULowerToRuntimeBase<
          FNGPULowerToRuntimePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    llvm::SmallVector<fir::fngpu::LaunchOp> launches;
    module.walk(
        [&](fir::fngpu::LaunchOp launchOp) { launches.push_back(launchOp); });

    int32_t fallbackKernelId = 0;

    for (fir::fngpu::LaunchOp launchOp : launches) {
      auto result = fir::fngpu::recognizeElementwiseKernel(launchOp);

      if (result.failed()) {
        launchOp.emitWarning("FNGPU runtime lowering skipped launch: ")
            << result.getFailure().reason;
        ++fallbackKernelId;
        continue;
      }

      const fir::fngpu::ElementwiseKernel &k = result.getKernel();

      Location loc = launchOp.getLoc();
      BlockShape blockShape = getBlockShape(launchOp);

      builder.setInsertionPoint(launchOp);

      int32_t stableKernelId = getKernelId(launchOp, fallbackKernelId);

      Value kernelIdValue =
          arith::ConstantIntOp::create(builder, loc, stableKernelId, 32);

      Value rankValue = arith::ConstantIntOp::create(builder, loc, k.rank, 32);

      Value blockXValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.x, 32);

      Value blockYValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.y, 32);

      Value blockZValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.z, 32);

      Value extentXValue = fir::LoadOp::create(builder, loc, k.nRef);

      Value extentYValue;
      if (k.rank == 2) {
        extentYValue = fir::LoadOp::create(builder, loc, k.mRef);
      } else {
        extentYValue = arith::ConstantIntOp::create(builder, loc, 1, 32);
      }

      Value extentZValue = arith::ConstantIntOp::create(builder, loc, 1, 32);

      // Use a single rank-independent runtime ABI.
      //
      // Do not declare the runtime using rank-specific array types such as:
      //
      //   !fir.ref<!fir.array<?xf32>>
      //   !fir.ref<!fir.array<?x?xf32>>
      //
      // Convert all arrays to !fir.ref<f32> before the call.
      StringRef runtimeName = "__fngpu_launch_f32_v1";

      FloatType f32Ty = builder.getF32Type();
      Type f32RefTy = fir::ReferenceType::get(f32Ty);

      Value read0Ptr =
          fir::ConvertOp::create(builder, loc, f32RefTy, k.readArrays[0]);

      Value read1Ptr;
      if (k.readArrays.size() >= 2) {
        read1Ptr =
            fir::ConvertOp::create(builder, loc, f32RefTy, k.readArrays[1]);
      } else {
        read1Ptr = read0Ptr;
      }

      Value read2Ptr = read0Ptr;

      Value writePtr =
          fir::ConvertOp::create(builder, loc, f32RefTy, k.writeArray);

      unsigned scalarCount = k.scalarRefs.size();

      if (k.readArrays.size() > 3) {
        launchOp.emitError(
            "FNGPU generic runtime supports at most three read arrays");
        signalPassFailure();
        return;
      }

      if (scalarCount > 3) {
        launchOp.emitError(
            "FNGPU generic runtime supports at most three f32 scalars");
        signalPassFailure();
        return;
      }

      Value numReadArraysValue =
          arith::ConstantIntOp::create(builder, loc, k.readArrays.size(), 32);

      Value numScalarsValue =
          arith::ConstantIntOp::create(builder, loc, scalarCount, 32);

      auto zeroAttr = builder.getFloatAttr(f32Ty, 0.0);

      Value zeroF32 = arith::ConstantOp::create(builder, loc, f32Ty, zeroAttr);

      Value scalar0Value = zeroF32;
      Value scalar1Value = zeroF32;
      Value scalar2Value = zeroF32;

      if (scalarCount >= 1)
        scalar0Value = fir::LoadOp::create(builder, loc, k.scalarRefs[0]);

      if (scalarCount >= 2)
        scalar1Value = fir::LoadOp::create(builder, loc, k.scalarRefs[1]);

      if (scalarCount >= 3)
        scalar2Value = fir::LoadOp::create(builder, loc, k.scalarRefs[2]);

      llvm::SmallVector<Type> argTypes;
      argTypes.push_back(kernelIdValue.getType());
      argTypes.push_back(rankValue.getType());
      argTypes.push_back(blockXValue.getType());
      argTypes.push_back(blockYValue.getType());
      argTypes.push_back(blockZValue.getType());
      argTypes.push_back(numReadArraysValue.getType());
      argTypes.push_back(numScalarsValue.getType());
      argTypes.push_back(f32RefTy);
      argTypes.push_back(f32RefTy);
      argTypes.push_back(f32RefTy);
      argTypes.push_back(f32RefTy);
      argTypes.push_back(f32Ty);
      argTypes.push_back(f32Ty);
      argTypes.push_back(f32Ty);
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
      operands.push_back(numReadArraysValue);
      operands.push_back(numScalarsValue);
      operands.push_back(read0Ptr);
      operands.push_back(read1Ptr);
      operands.push_back(read2Ptr);
      operands.push_back(writePtr);
      operands.push_back(scalar0Value);
      operands.push_back(scalar1Value);
      operands.push_back(scalar2Value);
      operands.push_back(extentXValue);
      operands.push_back(extentYValue);
      operands.push_back(extentZValue);

      func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                           operands);

      launchOp.erase();

      ++fallbackKernelId;
    }
    lowerFNGPUDataOpsToRuntime(module, builder);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPULowerToRuntimePass() {
  return std::make_unique<FNGPULowerToRuntimePass>();
}

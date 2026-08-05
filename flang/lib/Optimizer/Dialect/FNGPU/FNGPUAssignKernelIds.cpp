#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPUASSIGNKERNELIDS
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kKernelIdAttrName = "fngpu.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fngpu.kernel_name";

struct FNGPUAssignKernelIdsPass
    : public fir::fngpu::impl::FNGPUAssignKernelIdsBase<
          FNGPUAssignKernelIdsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder builder(ctx);

    int32_t nextId = 0;

    module.walk([&](fir::fngpu::LaunchOp launchOp) {
      int32_t id = nextId++;

      // If an id already exists, preserve it. This makes the pass idempotent
      // enough for debugging pipelines that accidentally run it twice.
      if (auto existingId =
              launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName)) {
        id = static_cast<int32_t>(existingId.getInt());
      } else {
        launchOp->setAttr(kKernelIdAttrName, builder.getI32IntegerAttr(id));
      }

      if (!launchOp->hasAttr(kKernelNameAttrName)) {
        std::string name = "fngpu_kernel_" + std::to_string(id);
        launchOp->setAttr(kKernelNameAttrName, builder.getStringAttr(name));
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPUAssignKernelIdsPass() {
  return std::make_unique<FNGPUAssignKernelIdsPass>();
}

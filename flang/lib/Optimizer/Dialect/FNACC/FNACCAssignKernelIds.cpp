#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCASSIGNKERNELIDS
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kKernelIdAttrName = "fnacc.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fnacc.kernel_name";

struct FNACCAssignKernelIdsPass
    : public fir::fnacc::impl::FNACCAssignKernelIdsBase<
          FNACCAssignKernelIdsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder builder(ctx);

    int32_t nextId = 0;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
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
        std::string name = "fnacc_kernel_" + std::to_string(id);
        launchOp->setAttr(kKernelNameAttrName, builder.getStringAttr(name));
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCAssignKernelIdsPass() {
  return std::make_unique<FNACCAssignKernelIdsPass>();
}

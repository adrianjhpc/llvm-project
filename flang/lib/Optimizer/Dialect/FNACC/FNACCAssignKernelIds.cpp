#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <limits>
#include <string>

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

    llvm::DenseSet<int32_t> usedIds;
    bool invalid = false;

    // Validate pre-existing IDs before assigning any new ones. In particular,
    // never let a preserved ID collide with a walk-order generated ID.
    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      auto existingId = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName);
      if (!existingId)
        return;

      int64_t rawId = existingId.getInt();
      if (rawId < 0 || rawId > std::numeric_limits<int32_t>::max()) {
        launchOp.emitError("FNACC kernel id is outside the i32 range");
        invalid = true;
        return;
      }

      int32_t id = static_cast<int32_t>(rawId);
      if (!usedIds.insert(id).second) {
        launchOp.emitError("duplicate FNACC kernel id ") << id;
        invalid = true;
      }

      if (auto function = launchOp->getParentOfType<mlir::func::FuncOp>()) {
        function->setAttr("fnacc.contains_launch", builder.getUnitAttr());
      }
    });

    if (invalid) {
      signalPassFailure();
      return;
    }

    int32_t nextId = 0;
    llvm::StringSet<> usedNames;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      auto existingId = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName);
      int32_t id = 0;

      if (existingId) {
        id = static_cast<int32_t>(existingId.getInt());
      } else {
        while (usedIds.count(nextId) != 0) {
          if (nextId == std::numeric_limits<int32_t>::max()) {
            launchOp.emitError("exhausted FNACC i32 kernel id space");
            invalid = true;
            return;
          }
          ++nextId;
        }
        id = nextId;
        if (nextId < std::numeric_limits<int32_t>::max())
          ++nextId;
        usedIds.insert(id);
        launchOp->setAttr(kKernelIdAttrName, builder.getI32IntegerAttr(id));
      }

      if (auto function = launchOp->getParentOfType<mlir::func::FuncOp>()) {
        function->setAttr("fnacc.contains_launch", builder.getUnitAttr());
      }

      if (!launchOp->hasAttr(kKernelNameAttrName)) {
        std::string name = "fnacc_kernel_" + std::to_string(id);
        launchOp->setAttr(kKernelNameAttrName, builder.getStringAttr(name));
      }

      StringRef name =
          launchOp->getAttrOfType<StringAttr>(kKernelNameAttrName).getValue();
      bool validName =
          !name.empty() && (llvm::isAlpha(name.front()) || name.front() == '_');
      for (char ch : name.drop_front())
        validName &= llvm::isAlnum(ch) || ch == '_';
      if (!validName) {
        launchOp.emitError(
            "FNACC kernel name must match [A-Za-z_][A-Za-z0-9_]*");
        invalid = true;
        return;
      }
      if (!usedNames.insert(name).second) {
        launchOp.emitError("duplicate FNACC kernel name '") << name << "'";
        invalid = true;
      }
    });

    if (invalid)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCAssignKernelIdsPass() {
  return std::make_unique<FNACCAssignKernelIdsPass>();
}

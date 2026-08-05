#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPUOUTLINEKERNELS
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {
struct FNGPUOutlineKernelsPass
    : public fir::fngpu::impl::FNGPUOutlineKernelsBase<
          FNGPUOutlineKernelsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(module.getContext());

    int kernelId = 0;

    // Walk the IR looking for fngpu.launch operations
    module.walk([&](fir::fngpu::LaunchOp launchOp) {
      outlineKernel(launchOp, builder, symbolTable, kernelId++);
    });
  }

private:
  void outlineKernel(fir::fngpu::LaunchOp launchOp, OpBuilder &builder,
                     SymbolTable &symbolTable, int kernelId) {
    Location loc = launchOp.getLoc();

    // 1. Variable Capturing
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(launchOp.getRegion(), captures);

    SmallVector<Type> argTypes;
    SmallVector<Value> callOperands;
    for (Value v : captures) {
      argTypes.push_back(v.getType());
      callOperands.push_back(v); // We will pass these from the host
    }

    // 2. Create the Kernel Function
    builder.setInsertionPointToEnd(
        launchOp->getParentOfType<ModuleOp>().getBody());

    FunctionType funcType = builder.getFunctionType(argTypes, TypeRange{});
    std::string kernelName = "fngpu_kernel_" + std::to_string(kernelId);

    // Create function using StringRef to avoid deprecation warning
    auto kernelFunc =
        builder.create<func::FuncOp>(loc, StringRef(kernelName), funcType);
    kernelFunc.setPrivate();
    symbolTable.insert(kernelFunc);

    // 3. Move the region body into the new function
    Region &kernelRegion = kernelFunc.getBody();
    kernelRegion.takeBody(launchOp.getRegion());

    // 4. Map captured values to new block arguments
    Block &entryBlock = kernelRegion.front();
    for (Value capture : captures) {
      // Add an argument to the kernel entry block for each captured variable
      BlockArgument arg = entryBlock.addArgument(capture.getType(), loc);

      // Replace all uses of the captured value *inside* the kernel with the
      // block argument
      capture.replaceUsesWithIf(arg, [&](OpOperand &operand) {
        return kernelRegion.isAncestor(operand.getOwner()->getParentRegion());
      });
    }

    // 5. Replace the original launch operation with a host call
    builder.setInsertionPoint(launchOp);
    builder.create<func::CallOp>(loc, kernelFunc, callOperands);
    launchOp.erase();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPUOutlineKernelsPass() {
  return std::make_unique<FNGPUOutlineKernelsPass>();
}

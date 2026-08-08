#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCOUTLINEKERNELS
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {
struct FNACCOutlineKernelsPass
    : public fir::fnacc::impl::FNACCOutlineKernelsBase<
          FNACCOutlineKernelsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    OpBuilder builder(module.getContext());

    int kernelId = 0;

    // Walk the IR looking for fnacc.launch operations
    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      outlineKernel(launchOp, builder, symbolTable, kernelId++);
    });
  }

private:
  void outlineKernel(fir::fnacc::LaunchOp launchOp, OpBuilder &builder,
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
    std::string kernelName = "fnacc_kernel_" + std::to_string(kernelId);

    // Create function using StringRef to avoid deprecation warning
    auto kernelFunc =
        builder.create<func::FuncOp>(loc, StringRef(kernelName), funcType);
    kernelFunc.setPrivate();

    // 3. Move the region body into the new function
    Region &kernelRegion = kernelFunc.getBody();
    kernelRegion.takeBody(launchOp.getRegion());

    // The launch region was terminated with fir.end. A func.func body must
    // terminate with func.return, so rewrite any fir.end terminators.
    for (auto endOp :
         llvm::make_early_inc_range(kernelFunc.getOps<fir::FirEndOp>())) {
      builder.setInsertionPoint(endOp);
      func::ReturnOp::create(builder, endOp.getLoc());
      endOp.erase();
    }

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

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCOutlineKernelsPass() {
  return std::make_unique<FNACCOutlineKernelsPass>();
}

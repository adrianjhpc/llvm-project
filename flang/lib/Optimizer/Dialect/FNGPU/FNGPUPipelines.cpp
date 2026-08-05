#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir::fngpu {
namespace {

struct FNGPUPipelineOptions
    : public mlir::PassPipelineOptions<FNGPUPipelineOptions> {
  Option<std::string> ttirOutput{
      *this, "ttir-output",
      llvm::cl::desc("Path to write generated FNGPU Triton TTIR"),
      llvm::cl::init("fngpu_kernels.ttir")};

  Option<std::string> jsonOutput{
      *this, "json-output",
      llvm::cl::desc("Path to write generated FNGPU kernel descriptor JSON"),
      llvm::cl::init("fngpu_kernels.json")};

  Option<bool> emitFortranAliases{
      *this, "emit-fortran-aliases",
      llvm::cl::desc("Emit external Fortran ABI aliases for transformed "
                     "top-level procedures"),
      llvm::cl::init(false)};
};

void buildFNGPUPipeline(mlir::OpPassManager &pm,
                        const FNGPUPipelineOptions &options) {
  pm.addPass(createFNGPUAssignKernelIdsPass());

  pm.addPass(
      createFNGPULowerToTritonPass(options.ttirOutput, options.jsonOutput));

  pm.addPass(createFNGPULowerToRuntimePass());

  if (options.emitFortranAliases)
    pm.addPass(createFNGPUEmitFortranAliasesPass());
}

} // namespace

void registerFNGPUPipelines() {
  mlir::PassPipelineRegistration<FNGPUPipelineOptions>(
      "fngpu-pipeline",
      "Run the experimental FNGPU lowering pipeline: assign kernel ids, emit "
      "Triton TTIR/JSON metadata, and lower host FNGPU operations to runtime "
      "calls",
      buildFNGPUPipeline);
}

} // namespace fir::fngpu

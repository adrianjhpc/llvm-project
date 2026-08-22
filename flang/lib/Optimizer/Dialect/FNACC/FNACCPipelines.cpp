#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir::fnacc {
namespace {

struct FNACCPipelineOptions
    : public mlir::PassPipelineOptions<FNACCPipelineOptions> {
  Option<std::string> ttirOutput{
      *this, "ttir-output",
      llvm::cl::desc("Path to write generated FNACC Triton TTIR"),
      llvm::cl::init("fnacc_kernels.ttir")};

  Option<std::string> jsonOutput{
      *this, "json-output",
      llvm::cl::desc("Path to write generated FNACC kernel descriptor JSON"),
      llvm::cl::init("fnacc_kernels.json")};

  Option<bool> emitFortranAliases{
      *this, "emit-fortran-aliases",
      llvm::cl::desc("Emit external Fortran ABI aliases for transformed "
                     "top-level procedures"),
      llvm::cl::init(false)};
  Option<int32_t> numWarps{*this, "num-warps",
                           llvm::cl::desc("Number of Triton warps per CTA"),
                           llvm::cl::init(1)};

  Option<int32_t> threadsPerWarp{
      *this, "threads-per-warp",
      llvm::cl::desc("Number of CUDA threads per warp"), llvm::cl::init(32)};

  Option<int32_t> numStages{*this, "num-stages",
                            llvm::cl::desc("Number of Triton pipeline stages"),
                            llvm::cl::init(3)};

  Option<std::string> f64MatmulStrategy{
      *this, "f64-matmul-strategy",
      llvm::cl::desc("Strategy for f64 matmul lowering: dot, reduce, or fma"),
      llvm::cl::init("reduce")};

};

void buildFNACCPipeline(mlir::OpPassManager &pm,
                        const FNACCPipelineOptions &options) {
  pm.addPass(createFNACCAssignKernelIdsPass());

  pm.addPass(createFNACCLowerToTritonPass(
      options.ttirOutput, options.jsonOutput, options.numWarps,
      options.threadsPerWarp, options.numStages, options.f64MatmulStrategy));

  pm.addPass(createFNACCLowerToRuntimePass());

  if (options.emitFortranAliases)
    pm.addPass(createFNACCEmitFortranAliasesPass());
}

} // namespace

void registerFNACCPipelines() {
  mlir::PassPipelineRegistration<FNACCPipelineOptions>(
      "fnacc-pipeline",
      "Run the experimental FNACC lowering pipeline: assign kernel ids, emit "
      "Triton TTIR/JSON metadata, and lower host FNACC operations to runtime "
      "calls",
      buildFNACCPipeline);
}

} // namespace fir::fnacc

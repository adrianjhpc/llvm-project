#include "flang/Optimizer/Dialect/TileOffload/TileOffloadPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir::TileOffload {
namespace {

struct TileOffloadPipelineOptions
    : public mlir::PassPipelineOptions<TileOffloadPipelineOptions> {
  Option<std::string> ttirOutput{
      *this, "ttir-output",
      llvm::cl::desc("Path to write generated TileOffload Triton TTIR"),
      llvm::cl::init("tileoff_kernels.ttir")};

  Option<std::string> jsonOutput{
      *this, "json-output",
      llvm::cl::desc("Path to write generated TileOffload kernel descriptor JSON"),
      llvm::cl::init("tileoff_kernels.json")};

  Option<bool> emitFortranAliases{
      *this, "emit-fortran-aliases",
      llvm::cl::desc("Emit external Fortran ABI aliases for transformed "
                     "top-level procedures"),
      llvm::cl::init(false)};
  Option<int32_t> launchAbi{*this, "launch-abi",
                            llvm::cl::desc("Host launch ABI: 2 or 3"),
                            llvm::cl::init(3)};

  Option<int32_t> numWarps{*this, "num-warps",
                           llvm::cl::desc("Number of Triton warps per CTA"),
                           llvm::cl::init(1)};

  Option<int32_t> threadsPerWarp{
      *this, "threads-per-warp",
      llvm::cl::desc("Number of threads per GPU subgroup"), llvm::cl::init(32)};

  Option<int32_t> numStages{*this, "num-stages",
                            llvm::cl::desc("Number of Triton pipeline stages"),
                            llvm::cl::init(3)};

  Option<std::string> f64MatmulStrategy{
      *this, "f64-matmul-strategy",
      llvm::cl::desc("Strategy for f64 matmul lowering: dot, reduce, or fma"),
      llvm::cl::init("reduce")};

  Option<std::string> acceleratorTarget{
      *this, "accelerator-target",
      llvm::cl::desc("Accelerator target for Triton lowering: cuda or hip"),
      llvm::cl::init("cuda")};

  Option<std::string> backend{
      *this, "backend",
      llvm::cl::desc("Preferred TileOffload device-code backend: auto or triton"),
      llvm::cl::init("auto")};

  Option<std::string> fallbackBackend{
      *this, "fallback-backend",
      llvm::cl::desc("TileOffload backend used when the preferred backend cannot "
                     "lower a kernel"),
      llvm::cl::init("triton")};

  Option<bool> allowBackendFallback{
      *this, "allow-backend-fallback",
      llvm::cl::desc("Allow per-kernel fallback to fallback-backend"),
      llvm::cl::init(true)};
};

void buildTileOffloadPipeline(mlir::OpPassManager &pm,
                        const TileOffloadPipelineOptions &options) {
  pm.addPass(createTileOffloadAssignKernelIdsPass());

  pm.addPass(createTileOffloadLowerToTritonPass(
      options.ttirOutput, options.jsonOutput, options.numWarps,
      options.threadsPerWarp, options.numStages, options.f64MatmulStrategy,
      options.backend, options.fallbackBackend, options.allowBackendFallback,
      options.acceleratorTarget));

  pm.addPass(createTileOffloadLowerToRuntimePass(options.launchAbi));

  if (options.emitFortranAliases)
    pm.addPass(createTileOffloadEmitFortranAliasesPass());
}

} // namespace

void registerTileOffloadPipelines() {
  mlir::PassPipelineRegistration<TileOffloadPipelineOptions>(
      "TileOffload-pipeline",
      "Run the experimental TileOffload lowering pipeline: assign kernel ids, emit "
      "Triton TTIR/JSON metadata, and lower host TileOffload operations to runtime "
      "calls",
      buildTileOffloadPipeline);
}

} // namespace fir::TileOffload

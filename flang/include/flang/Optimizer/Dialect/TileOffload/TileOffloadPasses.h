#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_H
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>

namespace fir::TileOffload {
#define GEN_PASS_DECL
#include "flang/Optimizer/Dialect/TileOffload/TileOffloadPasses.h.inc"

std::unique_ptr<mlir::Pass> createTileOffloadAssignKernelIdsPass();
std::unique_ptr<mlir::Pass> createTileOffloadOutlineKernelsPass();
std::unique_ptr<mlir::Pass> createTileOffloadEmitFortranAliasesPass();
std::unique_ptr<mlir::Pass> createTileOffloadLowerToTritonPass();
std::unique_ptr<mlir::Pass> createTileOffloadLowerToTritonPass(
    llvm::StringRef ttirOutput, llvm::StringRef jsonOutput, int32_t numWarps,
    int32_t threadsPerWarp, int32_t numStages,
    llvm::StringRef f64MatmulStrategy, llvm::StringRef backend = "auto",
    llvm::StringRef fallbackBackend = "triton",
    bool allowBackendFallback = true);
std::unique_ptr<mlir::Pass> createTileOffloadLowerToTritonPass(
    llvm::StringRef ttirOutput, llvm::StringRef jsonOutput, int32_t numWarps,
    int32_t threadsPerWarp, int32_t numStages,
    llvm::StringRef f64MatmulStrategy, llvm::StringRef backend,
    llvm::StringRef fallbackBackend, bool allowBackendFallback,
    llvm::StringRef acceleratorTarget);
std::unique_ptr<mlir::Pass> createTileOffloadLowerToRuntimePass();
std::unique_ptr<mlir::Pass> createTileOffloadLowerToRuntimePass(int32_t launchAbi);

void registerTileOffloadPipelines();

#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Dialect/TileOffload/TileOffloadPasses.h.inc"
} // namespace fir::TileOffload

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_H

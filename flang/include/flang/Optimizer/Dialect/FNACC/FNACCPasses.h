#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCPASSES_H
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCPASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace fir::fnacc {
#define GEN_PASS_DECL
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"

std::unique_ptr<mlir::Pass> createFNACCAssignKernelIdsPass();
std::unique_ptr<mlir::Pass> createFNACCOutlineKernelsPass();
std::unique_ptr<mlir::Pass> createFNACCEmitFortranAliasesPass();
std::unique_ptr<mlir::Pass> createFNACCLowerToTritonPass();
std::unique_ptr<mlir::Pass> createFNACCLowerToTritonPass(
    llvm::StringRef ttirOutput, llvm::StringRef jsonOutput, int32_t numWarps,
    int32_t threadsPerWarp, int32_t numStages,
    llvm::StringRef f64MatmulStrategy, llvm::StringRef backend = "auto",
    llvm::StringRef fallbackBackend = "triton",
    bool allowBackendFallback = true);
std::unique_ptr<mlir::Pass> createFNACCLowerToTritonPass(
    llvm::StringRef ttirOutput, llvm::StringRef jsonOutput, int32_t numWarps,
    int32_t threadsPerWarp, int32_t numStages,
    llvm::StringRef f64MatmulStrategy, llvm::StringRef backend,
    llvm::StringRef fallbackBackend, bool allowBackendFallback,
    llvm::StringRef acceleratorTarget);
std::unique_ptr<mlir::Pass> createFNACCLowerToRuntimePass();

void registerFNACCPipelines();

#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCPASSES_H

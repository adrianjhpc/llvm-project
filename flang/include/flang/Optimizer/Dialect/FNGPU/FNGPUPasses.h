#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUPASSES_H
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace fir::fngpu {
#define GEN_PASS_DECL
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"

std::unique_ptr<mlir::Pass> createFNGPUOutlineKernelsPass();
std::unique_ptr<mlir::Pass> createFNGPULowerToTritonPass();

#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUPASSES_H

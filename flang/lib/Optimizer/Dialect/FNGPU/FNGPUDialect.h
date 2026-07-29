#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H

#include "mlir/IR/Dialect.h"

// Pull in the auto-generated declarations for the Dialect
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h.inc"

// Pull in the auto-generated declarations for the Operations
#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FNGPU/FNGPUOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H

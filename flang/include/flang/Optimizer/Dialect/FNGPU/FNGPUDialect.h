#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The .inc files automatically generate the fir::fngpu namespaces
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FNGPU/FNGPUOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUDIALECT_H

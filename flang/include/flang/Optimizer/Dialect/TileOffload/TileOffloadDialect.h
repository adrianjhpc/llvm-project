#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The .inc files automatically generate the fir::TileOffload namespaces
#include "flang/Optimizer/Dialect/TileOffload/TileOffloadDialect.h.inc"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/TileOffload/TileOffloadOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_H

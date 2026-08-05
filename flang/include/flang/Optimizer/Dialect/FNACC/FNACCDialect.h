#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The .inc files automatically generate the fir::fnacc namespaces
#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h.inc"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FNACC/FNACCOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCDIALECT_H

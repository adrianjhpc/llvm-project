#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_DIALECT
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_DIALECT

include "mlir/IR/OpBase.td"

def FNACC_Dialect : Dialect {
  let name = "fnacc";
  let cppNamespace = "::fir::fnacc";
  let summary = "The Flang GPU directive dialect";
  let description = [{
    This dialect represents `$FNACC` pragmas in Flang before they are
    lowered to Triton or GPU execution models.
  }];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_DIALECT

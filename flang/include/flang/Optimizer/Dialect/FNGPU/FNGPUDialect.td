#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_DIALECT
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_DIALECT

include "mlir/IR/OpBase.td"

def FNGPU_Dialect : Dialect {
  let name = "fngpu";
  let cppNamespace = "::fir::fngpu";
  let summary = "The Flang GPU directive dialect";
  let description = [{
    This dialect represents `$FNGPU` pragmas in Flang before they are
    lowered to Triton or GPU execution models.
  }];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_DIALECT

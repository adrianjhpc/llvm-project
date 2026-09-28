#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_TD
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_TD

include "mlir/IR/OpBase.td"

def TileOffload_Dialect : Dialect {
  let name = "TileOffload";
  let cppNamespace = "::fir::TileOffload";
  let summary = "The Flang GPU directive dialect";
  let description = [{
    This dialect represents `$TileOff` pragmas in Flang before they are
    lowered to Triton or GPU execution models.
  }];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_DIALECT_TD

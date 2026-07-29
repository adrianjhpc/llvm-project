#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_OPS
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_OPS

include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all FNGPU operations
class FNGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<FNGPU_Dialect, mnemonic, traits>;

def FNGPU_LaunchOp : FNGPU_Op<"launch", [RecursiveMemoryEffects]> {
  let summary = "FNGPU launch operation";
  let description = [{
    The `fngpu.launch` operation represents a parallel execution region.
    It carries constant tile sizes and a list of packed variables, each
    tagged for host (0) or device (1) placement.
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$tile_sizes,
    Variadic<AnyType>:$pack_vars,       // SSA values for A, B, C (!fir.ref<...>)
    DenseI32ArrayAttr:$pack_targets     // 0 = Host, 1 = Device, one per pack_var
  );

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `tile_sizes` `=` $tile_sizes
    ( `pack` `(` $pack_vars^ `:` type($pack_vars) `)` )?
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}


#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_OPS

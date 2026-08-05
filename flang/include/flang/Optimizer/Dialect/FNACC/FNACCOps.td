#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_OPS
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_OPS

include "flang/Optimizer/Dialect/FNACC/FNACCDialect.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all FNACC operations
class FNACC_Op<string mnemonic, list<Trait> traits = []> :
    Op<FNACC_Dialect, mnemonic, traits>;

def FNACC_LaunchOp : FNACC_Op<"launch", [RecursiveMemoryEffects]> {
  let summary = "FNACC launch operation";
  let description = [{
    The `fnacc.launch` operation represents a parallel execution region.
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

def FNACC_UpdateHostOp : FNACC_Op<"update_host", []> {
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def FNACC_UpdateDeviceOp : FNACC_Op<"update_device", []> {
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def FNACC_ReleaseOp : FNACC_Op<"release", []> {
  let arguments = (ins Variadic<AnyType>:$vars);
  let assemblyFormat = "$vars attr-dict `:` type($vars)";
}

def FNACC_ReleaseAllOp : FNACC_Op<"release_all", []> {
  let summary = "Release all cached FNACC device allocations";
  let description = [{
    Releases every cached device allocation managed by the FNACC runtime.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_OPS

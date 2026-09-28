#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_OPS_TD
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_OPS_TD

include "flang/Optimizer/Dialect/TileOffload/TileOffloadDialect.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all TileOffload operations
class tileoff_Op<string mnemonic, list<Trait> traits = []> :
    Op<TileOffload_Dialect, mnemonic, traits>;

def tileoff_TerminatorOp
    : tileoff_Op<"terminator", [Pure, Terminator]> {
  let summary = "Terminate an TileOffload region";
  let description = [{
    The `TileOffload.terminator` operation terminates a single-block TileOffload region.
  }];

  let assemblyFormat = "attr-dict";
}

def tileoff_LaunchOp
    : tileoff_Op<"launch",
          [RecursiveMemoryEffects,
           SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "TileOffload launch operation";
  let description = [{
    The `TileOffload.launch` operation represents a parallel execution region.
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

def tileoff_UpdateHostOp : tileoff_Op<"update_host", []> {
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_UpdateDeviceOp : tileoff_Op<"update_device", []> {
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_PresentOp : tileoff_Op<"present", []> {
  let summary = "Assert that an object is present in the TileOffload device cache";
  let description = [{
    Checks that the object already has a sufficiently large cached device
    allocation. The operation does not allocate memory or transfer data.
  }];
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_ReleaseOp : tileoff_Op<"release", []> {
  let arguments = (ins Variadic<AnyType>:$vars);
  let assemblyFormat = "$vars attr-dict `:` type($vars)";
}

def tileoff_ReleaseAllOp : tileoff_Op<"release_all", []> {
  let summary = "Release all cached TileOffload device allocations";
  let description = [{
    Releases every cached device allocation managed by the TileOffload runtime.
  }];

  let assemblyFormat = "attr-dict";
}

def tileoff_WaitOp : tileoff_Op<"wait", []> {
  let summary = "Wait for preceding TileOffload work";
  let description = [{
    Waits until all work previously submitted to the TileOffload runtime stream for
    the active CUDA context has completed. TileOffload currently has one stream per
    context, so this operation has no queue operands.
  }];

  let assemblyFormat = "attr-dict";
}

def tileoff_DataRegionEnterOp : tileoff_Op<"data_region_enter", []> {
  let summary = "Begin a nested TileOffload data ownership region";
  let description = [{
    Begins a data-region frame. COPYIN and CREATE operations following this
    marker acquire ownership in this frame without disturbing allocations
    owned by an enclosing frame.
  }];
  let assemblyFormat = "attr-dict";
}

def tileoff_DataRegionExitOp : tileoff_Op<"data_region_exit", []> {
  let summary = "End the current nested TileOffload data ownership region";
  let description = [{
    Ends the innermost data-region frame and releases the ownership acquired
    by that frame. Allocations still referenced by enclosing frames remain
    resident on the device.
  }];
  let assemblyFormat = "attr-dict";
}

def tileoff_CopyinOp : tileoff_Op<"copyin", []> {
  let summary = "Begin TileOffload device lifetime and copy host data to device";
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_CreateOp : tileoff_Op<"create", []> {
  let summary = "Begin TileOffload device lifetime without copying host data";
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_CopyoutOp : tileoff_Op<"copyout", []> {
  let summary = "Copy final-owner TileOffload device data back to host";
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

def tileoff_DeleteOp : tileoff_Op<"delete", []> {
  let summary = "Mark one allocation for release at data-region exit";
  let arguments = (ins AnyType:$var);
  let assemblyFormat = "$var attr-dict `:` type($var)";
}

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_OPS_TD

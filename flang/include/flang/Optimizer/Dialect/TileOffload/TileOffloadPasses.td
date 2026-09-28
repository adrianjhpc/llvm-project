#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_TD
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_TD

include "mlir/Pass/PassBase.td"

def TileOffloadAssignKernelIds : Pass<"TileOffload-assign-kernel-ids", "mlir::ModuleOp"> {
  let summary = "Assign stable kernel ids and names to TileOffload.launch operations";
  let description = [{
    Walks TileOffload.launch operations and attaches stable `TileOffload.kernel_id`
    and `TileOffload.kernel_name` attributes. Later TileOffload passes use these
    attributes to keep generated device kernels and host runtime calls
    consistent across TileOffload backends.
  }];
  let constructor = "fir::TileOffload::createTileOffloadAssignKernelIdsPass()";
  let dependentDialects = [
    "fir::TileOffload::TileOffloadDialect"
  ];
}

def TileOffloadOutlineKernels : Pass<"TileOffload-outline-kernels", "mlir::ModuleOp"> {
  let summary = "Outlines TileOffload.launch regions into standalone kernel functions";
  let description = [{
    This pass finds `TileOffload.launch` operations, extracts their nested regions
    into standalone `func.func` operations at the module level, and replaces
    the original region with a call/launch operation.
  }];
  let constructor = "fir::TileOffload::createTileOffloadOutlineKernelsPass()";
  let dependentDialects = [
    "fir::TileOffload::TileOffloadDialect",
    "mlir::func::FuncDialect"
  ];
}

def TileOffloadEmitFortranAliases
    : Pass<"TileOffload-emit-fortran-aliases", "mlir::ModuleOp"> {
  let summary = "Emit external Fortran ABI aliases for top-level TileOffload procedures";
  let description = [{
    Creates wrapper functions such as `compute_foo_` for internal Flang
    top-level procedure symbols such as `_QPcompute_foo`. This removes the
    need for a manually maintained C++ alias source file when a separately
    compiled Fortran main calls transformed TileOffload procedures.
  }];
  let constructor = "fir::TileOffload::createTileOffloadEmitFortranAliasesPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def TileOffloadLowerToTriton : Pass<"TileOffload-lower-to-triton", "mlir::ModuleOp"> {
  let summary = "Plan TileOffload kernels and emit them with the Triton backend";
  let constructor = "fir::TileOffload::createTileOffloadLowerToTritonPass()";

  let options = [
    Option<"ttirOutput", "ttir-output", "std::string",
           "\"tileoff_kernels.ttir\"",
           "Path to write generated Triton TTIR">,
    Option<"jsonOutput", "json-output", "std::string",
           "\"tileoff_kernels.json\"",
           "Path to write generated TileOffload kernel descriptor JSON">,
    Option<"numWarps", "num-warps", "int32_t",
           "1",
           "Number of Triton warps per CTA">,
    Option<"threadsPerWarp", "threads-per-warp", "int32_t",
           "32",
           "Number of threads per GPU subgroup">,
    Option<"numStages", "num-stages", "int32_t",
           "3",
           "Number of Triton pipeline stages">,
    Option<"f64MatmulStrategy", "f64-matmul-strategy", "std::string",
           "\"reduce\"",
           "Strategy for f64 matmul lowering: dot, reduce, or fma">,
    Option<"acceleratorTarget", "accelerator-target", "std::string",
           "\"cuda\"",
           "Accelerator target for Triton lowering: cuda or hip">,
    Option<"backend", "backend", "std::string",
           "\"auto\"",
           "Preferred TileOffload device-code backend: auto or triton">,
    Option<"fallbackBackend", "fallback-backend", "std::string",
           "\"triton\"",
           "Backend used when the preferred backend cannot lower a kernel">,
    Option<"allowBackendFallback", "allow-backend-fallback", "bool",
           "true",
           "Allow per-kernel fallback to fallback-backend">
  ];

  let dependentDialects = [
    "fir::TileOffload::TileOffloadDialect",
    "fir::FIROpsDialect",
    "mlir::arith::ArithDialect"
  ];
}

def TileOffloadLowerToRuntime : Pass<"TileOffload-lower-to-runtime", "mlir::ModuleOp"> {
  let summary = "Lower TileOffload.launch operations to generated runtime wrapper calls";
  let description = [{
    Replaces recognized `TileOffload.launch` operations with calls to generated
    runtime wrappers such as `__tileoff_launch_nd_f32`.
  }];
  let constructor = "fir::TileOffload::createTileOffloadLowerToRuntimePass()";
  let options = [
    Option<"launchAbi", "launch-abi", "int32_t", "3",
           "Host launch ABI: 2 (binding calls) or 3 (single request)">
  ];
  let dependentDialects = [
    "fir::TileOffload::TileOffloadDialect",
    "fir::FIROpsDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_PASSES_TD

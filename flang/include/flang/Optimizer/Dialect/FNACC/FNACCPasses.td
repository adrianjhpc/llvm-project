#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_PASSES
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_PASSES

include "mlir/Pass/PassBase.td"

def FNACCAssignKernelIds : Pass<"fnacc-assign-kernel-ids", "mlir::ModuleOp"> {
  let summary = "Assign stable kernel ids and names to fnacc.launch operations";
  let description = [{
    Walks fnacc.launch operations and attaches stable `fnacc.kernel_id`
    and `fnacc.kernel_name` attributes. Later FNACC passes use these
    attributes to keep generated device kernels and host runtime calls
    consistent across FNACC backends.
  }];
  let constructor = "fir::fnacc::createFNACCAssignKernelIdsPass()";
  let dependentDialects = [
    "fir::fnacc::FNACCDialect"
  ];
}

def FNACCOutlineKernels : Pass<"fnacc-outline-kernels", "mlir::ModuleOp"> {
  let summary = "Outlines fnacc.launch regions into standalone kernel functions";
  let description = [{
    This pass finds `fnacc.launch` operations, extracts their nested regions
    into standalone `func.func` operations at the module level, and replaces
    the original region with a call/launch operation.
  }];
  let constructor = "fir::fnacc::createFNACCOutlineKernelsPass()";
  let dependentDialects = [
    "fir::fnacc::FNACCDialect",
    "mlir::func::FuncDialect"
  ];
}

def FNACCEmitFortranAliases
    : Pass<"fnacc-emit-fortran-aliases", "mlir::ModuleOp"> {
  let summary = "Emit external Fortran ABI aliases for top-level FNACC procedures";
  let description = [{
    Creates wrapper functions such as `compute_foo_` for internal Flang
    top-level procedure symbols such as `_QPcompute_foo`. This removes the
    need for a manually maintained C++ alias source file when a separately
    compiled Fortran main calls transformed FNACC procedures.
  }];
  let constructor = "fir::fnacc::createFNACCEmitFortranAliasesPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def FNACCLowerToTriton : Pass<"fnacc-lower-to-triton", "mlir::ModuleOp"> {
  let summary = "Plan FNACC kernels and emit them with the Triton backend";
  let constructor = "fir::fnacc::createFNACCLowerToTritonPass()";

  let options = [
    Option<"ttirOutput", "ttir-output", "std::string",
           "\"fnacc_kernels.ttir\"",
           "Path to write generated Triton TTIR">,
    Option<"jsonOutput", "json-output", "std::string",
           "\"fnacc_kernels.json\"",
           "Path to write generated FNACC kernel descriptor JSON">,
    Option<"numWarps", "num-warps", "int32_t",
           "1",
           "Number of Triton warps per CTA">,
    Option<"threadsPerWarp", "threads-per-warp", "int32_t",
           "32",
           "Number of CUDA threads per warp">,
    Option<"numStages", "num-stages", "int32_t",
           "3",
           "Number of Triton pipeline stages">,
    Option<"f64MatmulStrategy", "f64-matmul-strategy", "std::string",
           "\"reduce\"",
           "Strategy for f64 matmul lowering: dot, reduce, or fma">,
    Option<"backend", "backend", "std::string",
           "\"auto\"",
           "Preferred FNACC device-code backend: auto or triton">,
    Option<"fallbackBackend", "fallback-backend", "std::string",
           "\"triton\"",
           "Backend used when the preferred backend cannot lower a kernel">,
    Option<"allowBackendFallback", "allow-backend-fallback", "bool",
           "true",
           "Allow per-kernel fallback to fallback-backend">
  ];

  let dependentDialects = [
    "fir::fnacc::FNACCDialect",
    "fir::FIROpsDialect",
    "mlir::arith::ArithDialect"
  ];
}

def FNACCLowerToRuntime : Pass<"fnacc-lower-to-runtime", "mlir::ModuleOp"> {
  let summary = "Lower fnacc.launch operations to generated runtime wrapper calls";
  let description = [{
    Replaces recognized `fnacc.launch` operations with calls to generated
    runtime wrappers such as `__fnacc_launch_nd_f32`.
  }];
  let constructor = "fir::fnacc::createFNACCLowerToRuntimePass()";
  let dependentDialects = [
    "fir::fnacc::FNACCDialect",
    "fir::FIROpsDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_PASSES

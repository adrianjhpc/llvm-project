#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_PASSES
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_PASSES

include "mlir/Pass/PassBase.td"

def FNGPUAssignKernelIds : Pass<"fngpu-assign-kernel-ids", "mlir::ModuleOp"> {
  let summary = "Assign stable kernel ids and names to fngpu.launch operations";
  let description = [{
    Walks fngpu.launch operations and attaches stable `fngpu.kernel_id`
    and `fngpu.kernel_name` attributes. Later FNGPU passes use these
    attributes to keep generated Triton kernels and host runtime calls
    consistent.
  }];
  let constructor = "fir::fngpu::createFNGPUAssignKernelIdsPass()";
  let dependentDialects = [
    "fir::fngpu::FNGPUDialect"
  ];
}

def FNGPUOutlineKernels : Pass<"fngpu-outline-kernels", "mlir::ModuleOp"> {
  let summary = "Outlines fngpu.launch regions into standalone kernel functions";
  let description = [{
    This pass finds `fngpu.launch` operations, extracts their nested regions
    into standalone `func.func` operations at the module level, and replaces
    the original region with a call/launch operation.
  }];
  let constructor = "fir::fngpu::createFNGPUOutlineKernelsPass()";
  let dependentDialects = [
    "fir::fngpu::FNGPUDialect",
    "mlir::func::FuncDialect"
  ];
}

def FNGPUEmitFortranAliases
    : Pass<"fngpu-emit-fortran-aliases", "mlir::ModuleOp"> {
  let summary = "Emit external Fortran ABI aliases for top-level FNGPU procedures";
  let description = [{
    Creates wrapper functions such as `compute_foo_` for internal Flang
    top-level procedure symbols such as `_QPcompute_foo`. This removes the
    need for a manually maintained C++ alias source file when a separately
    compiled Fortran main calls transformed FNGPU procedures.
  }];
  let constructor = "fir::fngpu::createFNGPUEmitFortranAliasesPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}


def FNGPULowerToTriton : Pass<"fngpu-lower-to-triton", "mlir::ModuleOp"> {
  let summary = "Emit Triton textual IR for elementwise fngpu.launch regions";
  let constructor = "fir::fngpu::createFNGPULowerToTritonPass()";

  let options = [
    Option<"ttirOutput", "ttir-output", "std::string",
           "\"fngpu_kernels.ttir\"",
           "Path to write generated Triton TTIR">,
    Option<"jsonOutput", "json-output", "std::string",
           "\"fngpu_kernels.json\"",
           "Path to write generated FNGPU kernel descriptor JSON">
  ];

  let dependentDialects = [
    "fir::fngpu::FNGPUDialect",
    "fir::FIROpsDialect",
    "mlir::arith::ArithDialect"
  ];
}

def FNGPULowerToRuntime : Pass<"fngpu-lower-to-runtime", "mlir::ModuleOp"> {
  let summary = "Lower fngpu.launch operations to generated runtime wrapper calls";
  let description = [{
    Replaces recognized `fngpu.launch` operations with calls to generated
    runtime wrappers such as `__fngpu_launch_nd_f32`.
  }];
  let constructor = "fir::fngpu::createFNGPULowerToRuntimePass()";
  let dependentDialects = [
    "fir::fngpu::FNGPUDialect",
    "fir::FIROpsDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_PASSES


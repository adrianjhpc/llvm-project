#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_PASSES
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_PASSES

include "mlir/Pass/PassBase.td"

def FNGPUOutlineKernels : Pass<"fngpu-outline-kernels", "mlir::ModuleOp"> {
  let summary = "Outlines fngpu.launch regions into standalone kernel functions";
  let description = [{
    This pass finds `fngpu.launch` operations, extracts their nested regions
    into standalone `func.func` (or `tt.func`) operations at the module level,
    and replaces the original region with a call/launch operation.
  }];
  let constructor = "fir::fngpu::createFNGPUOutlineKernelsPass()";
  let dependentDialects = [
    "fir::fngpu::FNGPUDialect",
    "mlir::func::FuncDialect"
  ];
}

def FNGPULowerToTriton : Pass<"fngpu-lower-to-triton", "mlir::ModuleOp"> {
  let summary = "Emit Triton textual IR for elementwise fngpu.launch regions";
  let constructor = "fir::fngpu::createFNGPULowerToTritonPass()";
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
    Fortran-callable C runtime wrappers such as `fngpu_launch_kernel_0`.
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

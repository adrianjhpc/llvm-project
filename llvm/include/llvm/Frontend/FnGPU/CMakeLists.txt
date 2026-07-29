set(LLVM_TARGET_DEFINITIONS FnGPU.td)
tablegen(LLVM FnGPU.h.inc -gen-directive-decl)
add_public_tablegen_target(FnGPUGen)

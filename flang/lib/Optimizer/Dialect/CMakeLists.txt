add_subdirectory(Support)
add_subdirectory(CUF)
add_subdirectory(FIRCG)
add_subdirectory(MIF)

add_flang_library(FIRDialect
  CUDAKernelOpInterface.cpp
  FIRAttr.cpp
  FIRBoxUtils.cpp
  FIRDialect.cpp
  FIROperationMoveOpInterface.cpp
  FIROps.cpp
  FIRType.cpp
  FirAliasTagOpInterface.cpp
  FortranVariableInterface.cpp
  Inliner.cpp
  SafeTempArrayCopyAttrInterface.cpp

  DEPENDS
  CanonicalizationPatternsIncGen
  CUDAKernelOpInterfaceIncGen
  FIROperationMoveOpInterfaceIncGen
  FIROpsIncGen
  FIRSafeTempArrayCopyAttrInterfaceIncGen
  CUFAttrsIncGen
  intrinsics_gen

  LINK_LIBS
  CUFAttrs
  FIRDialectSupport

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks

  MLIR_DEPS
  MLIRIR

  MLIR_LIBS
  MLIRArithDialect
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRBuiltinToLLVMIRTranslation
  MLIROpenMPToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
)

# --- FNACC Dialect Library Target ---
add_flang_library(FIRFNACCDialect
  FNACC/FNACCDialect.cpp
  FNACC/FNACCAssignKernelIds.cpp
  FNACC/FNACCEmitFortranAliases.cpp
  FNACC/FNACCKernelAnalysis.cpp
  FNACC/FNACCOutlineKernels.cpp
  FNACC/FNACCPipelines.cpp
  FNACC/FNACCLowerToTriton.cpp
  FNACC/FNACCLowerToRuntime.cpp

  DEPENDS
  FnACCDialectIncGen
  FNACCPassesIncGen

  LINK_LIBS
  FIRDialect 

  MLIR_DEPS
  MLIRIR
  MLIRPass
  MLIR_LIBS
  MLIRFuncDialect
  MLIRArithDialect 
  MLIRSupport
  MLIRTransformUtils
)

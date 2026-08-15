find_package(CUDAToolkit QUIET)

if(TARGET CUDA::cuda_driver)
  set(FNACC_CUDA_DRIVER_TARGET CUDA::cuda_driver)
else()
  find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h)
  find_library(CUDA_DRIVER_LIBRARY cuda)
  if(NOT CUDA_DRIVER_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
      "FLANG_FNACC_RUNTIME is ON, but the CUDA driver headers/library were "
      "not found. Set CUDAToolkit_ROOT (preferred), CUDA_DRIVER_INCLUDE_DIR "
      "and CUDA_DRIVER_LIBRARY, or disable FLANG_FNACC_RUNTIME.")
  endif()
  add_library(FNACCCudaDriver UNKNOWN IMPORTED)
  set_target_properties(FNACCCudaDriver PROPERTIES
    IMPORTED_LOCATION "${CUDA_DRIVER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_DRIVER_INCLUDE_DIR}")
  set(FNACC_CUDA_DRIVER_TARGET FNACCCudaDriver)
endif()

add_flang_library(FortranFNACCRuntime
  fnacc_runtime.cpp
)

target_link_libraries(FortranFNACCRuntime
  PRIVATE
    ${FNACC_CUDA_DRIVER_TARGET}
)


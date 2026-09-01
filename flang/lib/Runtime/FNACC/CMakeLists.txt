set(FLANG_FNACC_RUNTIME_BACKEND "CUDA" CACHE STRING
  "FnACC runtime backend(s): CUDA, HIP, or BOTH")
set_property(CACHE FLANG_FNACC_RUNTIME_BACKEND PROPERTY STRINGS CUDA HIP BOTH)
string(TOUPPER "${FLANG_FNACC_RUNTIME_BACKEND}" FNACC_RUNTIME_BACKEND)

if(NOT FNACC_RUNTIME_BACKEND MATCHES "^(CUDA|HIP|BOTH)$")
  message(FATAL_ERROR
    "FLANG_FNACC_RUNTIME_BACKEND must be CUDA, HIP, or BOTH (got "
    "'${FLANG_FNACC_RUNTIME_BACKEND}')")
endif()

if(FNACC_RUNTIME_BACKEND STREQUAL "CUDA" OR
   FNACC_RUNTIME_BACKEND STREQUAL "BOTH")
  find_package(CUDAToolkit QUIET)

  if(TARGET CUDA::cuda_driver)
    set(FNACC_CUDA_DRIVER_TARGET CUDA::cuda_driver)
  else()
    find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h)
    find_library(CUDA_DRIVER_LIBRARY cuda)
    if(NOT CUDA_DRIVER_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
      message(FATAL_ERROR
        "CUDA FnACC runtime requested, but the CUDA driver headers/library "
        "were not found. Set CUDAToolkit_ROOT, CUDA_DRIVER_INCLUDE_DIR and "
        "CUDA_DRIVER_LIBRARY, or select FLANG_FNACC_RUNTIME_BACKEND=HIP.")
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
  target_link_libraries(FortranFNACCRuntime PRIVATE
    ${FNACC_CUDA_DRIVER_TARGET})
endif()

if(FNACC_RUNTIME_BACKEND STREQUAL "HIP" OR
   FNACC_RUNTIME_BACKEND STREQUAL "BOTH")
  find_package(hip CONFIG QUIET)
  if(TARGET hip::host)
    set(FNACC_HIP_DRIVER_TARGET hip::host)
  else()
    find_path(HIP_DRIVER_INCLUDE_DIR hip/hip_runtime_api.h
      HINTS ENV ROCM_PATH ENV HIP_PATH
      PATH_SUFFIXES include)
    find_library(HIP_DRIVER_LIBRARY amdhip64
      HINTS ENV ROCM_PATH ENV HIP_PATH
      PATH_SUFFIXES lib lib64)
    if(NOT HIP_DRIVER_INCLUDE_DIR OR NOT HIP_DRIVER_LIBRARY)
      message(FATAL_ERROR
        "HIP FnACC runtime requested, but hip/hip_runtime_api.h or "
        "libamdhip64 was not found. Set ROCM_PATH/HIP_PATH, "
        "HIP_DRIVER_INCLUDE_DIR and HIP_DRIVER_LIBRARY, or select "
        "FLANG_FNACC_RUNTIME_BACKEND=CUDA.")
    endif()
    add_library(FNACCHipDriver UNKNOWN IMPORTED)
    set_target_properties(FNACCHipDriver PROPERTIES
      IMPORTED_LOCATION "${HIP_DRIVER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${HIP_DRIVER_INCLUDE_DIR}")
    set(FNACC_HIP_DRIVER_TARGET FNACCHipDriver)
  endif()

  # A distinct name lets CUDA and HIP runtimes coexist in one installation.
  # fnacc-flang selects the library matching its accelerator target.
  add_flang_library(FortranFNACCRuntimeHIP
    fnacc_runtime.cpp
  )
  target_compile_definitions(FortranFNACCRuntimeHIP PRIVATE
    FNACC_RUNTIME_USE_HIP=1)
  target_link_libraries(FortranFNACCRuntimeHIP PRIVATE
    ${FNACC_HIP_DRIVER_TARGET})
endif()

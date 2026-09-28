set(FLANG_TILEOFFLOAD_RUNTIME_BACKEND "CUDA" CACHE STRING
  "TileOffload runtime backend(s): CUDA, HIP, or BOTH")
set_property(CACHE FLANG_TILEOFFLOAD_RUNTIME_BACKEND PROPERTY STRINGS CUDA HIP BOTH)
string(TOUPPER "${FLANG_TILEOFFLOAD_RUNTIME_BACKEND}" TILEOFFLOAD_RUNTIME_BACKEND)

if(NOT TILEOFFLOAD_RUNTIME_BACKEND MATCHES "^(CUDA|HIP|BOTH)$")
  message(FATAL_ERROR
    "FLANG_TILEOFFLOAD_RUNTIME_BACKEND must be CUDA, HIP, or BOTH (got "
    "'${FLANG_TILEOFFLOAD_RUNTIME_BACKEND}')")
endif()

option(FLANG_TILEOFFLOAD_ENABLE_NVTX
  "Enable NVTX profiling in the TileOffload runtime" OFF)
set(TILEOFFLOAD_NVTX_INCLUDE_DIR "" CACHE PATH
  "Directory containing nvtx3/nvtx3.hpp (required when NVTX profiling is enabled)")

if(FLANG_TILEOFFLOAD_ENABLE_NVTX AND
   NOT EXISTS "${TILEOFFLOAD_NVTX_INCLUDE_DIR}/nvtx3/nvtx3.hpp")
  message(FATAL_ERROR
    "NVTX profiling is enabled. Set TILEOFFLOAD_NVTX_INCLUDE_DIR to the "
    "directory containing nvtx3/nvtx3.hpp, or set "
    "FLANG_TILEOFFLOAD_ENABLE_NVTX=OFF.")
endif()

function(tileoffload_configure_runtime target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  if(FLANG_TILEOFFLOAD_ENABLE_NVTX)
    target_compile_definitions(${target} PRIVATE TILEOFFLOAD_ENABLE_NVTX=1)
    target_include_directories(${target} PRIVATE
      "${TILEOFFLOAD_NVTX_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()
endfunction()

if(TILEOFFLOAD_RUNTIME_BACKEND STREQUAL "CUDA" OR
   TILEOFFLOAD_RUNTIME_BACKEND STREQUAL "BOTH")
  find_package(CUDAToolkit QUIET)

  if(TARGET CUDA::cuda_driver)
    set(TILEOFFLOAD_CUDA_DRIVER_TARGET CUDA::cuda_driver)
  else()
    find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h)
    find_library(CUDA_DRIVER_LIBRARY cuda)
    if(NOT CUDA_DRIVER_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
      message(FATAL_ERROR
        "CUDA TileOffload runtime requested, but the CUDA driver headers/library "
        "were not found. Set CUDAToolkit_ROOT, CUDA_DRIVER_INCLUDE_DIR and "
        "CUDA_DRIVER_LIBRARY, or select FLANG_TILEOFFLOAD_RUNTIME_BACKEND=HIP.")
    endif()
    add_library(TileOffloadCudaDriver UNKNOWN IMPORTED)
    set_target_properties(TileOffloadCudaDriver PROPERTIES
      IMPORTED_LOCATION "${CUDA_DRIVER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_DRIVER_INCLUDE_DIR}")
    set(TILEOFFLOAD_CUDA_DRIVER_TARGET TileOffloadCudaDriver)
  endif()

  add_flang_library(FortranTileOffloadRuntime
    tileoffload_runtime.cpp
  )
  tileoffload_configure_runtime(FortranTileOffloadRuntime)
  target_link_libraries(FortranTileOffloadRuntime PRIVATE
    ${TILEOFFLOAD_CUDA_DRIVER_TARGET})
endif()

if(TILEOFFLOAD_RUNTIME_BACKEND STREQUAL "HIP" OR
   TILEOFFLOAD_RUNTIME_BACKEND STREQUAL "BOTH")
  find_package(hip CONFIG QUIET)
  if(TARGET hip::host)
    set(TILEOFFLOAD_HIP_DRIVER_TARGET hip::host)
  else()
    find_path(HIP_DRIVER_INCLUDE_DIR hip/hip_runtime_api.h
      HINTS ENV ROCM_PATH ENV HIP_PATH
      PATH_SUFFIXES include)
    find_library(HIP_DRIVER_LIBRARY amdhip64
      HINTS ENV ROCM_PATH ENV HIP_PATH
      PATH_SUFFIXES lib lib64)
    if(NOT HIP_DRIVER_INCLUDE_DIR OR NOT HIP_DRIVER_LIBRARY)
      message(FATAL_ERROR
        "HIP TileOffload runtime requested, but hip/hip_runtime_api.h or "
        "libamdhip64 was not found. Set ROCM_PATH/HIP_PATH, "
        "HIP_DRIVER_INCLUDE_DIR and HIP_DRIVER_LIBRARY, or select "
        "FLANG_TILEOFFLOAD_RUNTIME_BACKEND=CUDA.")
    endif()
    add_library(TileOffloadHipDriver UNKNOWN IMPORTED)
    set_target_properties(TileOffloadHipDriver PROPERTIES
      IMPORTED_LOCATION "${HIP_DRIVER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${HIP_DRIVER_INCLUDE_DIR}")
    set(TILEOFFLOAD_HIP_DRIVER_TARGET TileOffloadHipDriver)
  endif()

  # A distinct name lets CUDA and HIP runtimes coexist in one installation.
  # TileOffload-flang selects the library matching its accelerator target.
  add_flang_library(FortranTileOffloadRuntimeHIP
    tileoffload_runtime.cpp
  )
  tileoffload_configure_runtime(FortranTileOffloadRuntimeHIP)
  target_compile_definitions(FortranTileOffloadRuntimeHIP PRIVATE
    TILEOFFLOAD_RUNTIME_USE_HIP=1)
  target_link_libraries(FortranTileOffloadRuntimeHIP PRIVATE
    ${TILEOFFLOAD_HIP_DRIVER_TARGET})
endif()

find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h
  PATHS /usr/local/cuda/include
)

find_library(CUDA_DRIVER_LIBRARY cuda
  PATHS /usr/local/cuda/lib64
)

if(NOT CUDA_DRIVER_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
  message(FATAL_ERROR
    "FLANG_FNACC_RUNTIME is ON, but the CUDA driver "
    "headers/library were not found. Set CUDA_DRIVER_INCLUDE_DIR and "
    "CUDA_DRIVER_LIBRARY, or disable FLANG_FNACC_RUNTIME.")
endif()

add_flang_library(FortranFNACCRuntime
  fnacc_runtime.cpp
)

target_include_directories(FortranFNACCRuntime
  PRIVATE
    ${CUDA_DRIVER_INCLUDE_DIR}
)

target_link_libraries(FortranFNACCRuntime
  PRIVATE
    ${CUDA_DRIVER_LIBRARY}
)


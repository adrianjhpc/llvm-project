find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h
  PATHS /usr/local/cuda/include
)

find_library(CUDA_DRIVER_LIBRARY cuda
  PATHS /usr/local/cuda/lib64
)

add_flang_library(FortranFNGPURuntime
  fngpu_runtime.cpp
)

target_include_directories(FortranFNGPURuntime
  PRIVATE
  ${CUDA_DRIVER_INCLUDE_DIR}
)

target_link_libraries(FortranFNGPURuntime
  PRIVATE
  ${CUDA_DRIVER_LIBRARY}
)


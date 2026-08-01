#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

#define FNGPU_CUDA_CHECK(call)						\
  do {									\
    CUresult _status = (call);						\
    if (_status != CUDA_SUCCESS) {					\
      const char *errName = nullptr;					\
      const char *errStr = nullptr;					\
      cuGetErrorName(_status, &errName);				\
      cuGetErrorString(_status, &errStr);				\
      std::fprintf(stderr,						\
                   "FNGPU CUDA driver error at %s:%d: %s: %s\n",	\
                   __FILE__,						\
                   __LINE__,						\
                   errName ? errName : "<unknown>",			\
                   errStr ? errStr : "<no details>");			\
      std::abort();							\
    }									\
  } while (0)

namespace {

  bool initialized = false;
  CUcontext context = nullptr;
  CUmodule module = nullptr;

  // kernelId -> CUfunction
  std::unordered_map<int32_t, CUfunction> functionCache;

  const char *getPtxPath() {
    // Let tests/users override the PTX path without recompiling.
    if (const char *env = std::getenv("FNGPU_PTX")) {
      return env;
    }

    // Prototype default: expect PTX in current working directory.
    return "fngpu_kernels.ptx";
  }

  void initializeRuntime() {
    if (initialized)
      return;

    FNGPU_CUDA_CHECK(cuInit(0));

    CUdevice device;
    FNGPU_CUDA_CHECK(cuDeviceGet(&device, 0));

    // Use the primary context to play nicely with other CUDA users in the process.
    FNGPU_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    FNGPU_CUDA_CHECK(cuCtxSetCurrent(context));

    const char *ptxPath = getPtxPath();

    FNGPU_CUDA_CHECK(cuModuleLoad(&module, ptxPath));

    initialized = true;
  }

  CUfunction getKernelFunction(int32_t kernelId) {
    initializeRuntime();

    auto it = functionCache.find(kernelId);
    if (it != functionCache.end())
      return it->second;

    std::string kernelName = "fngpu_kernel_" + std::to_string(kernelId);

    CUfunction fn = nullptr;
    FNGPU_CUDA_CHECK(cuModuleGetFunction(&fn, module, kernelName.c_str()));

    functionCache.emplace(kernelId, fn);
    return fn;
  }

  int32_t ceilDiv(int32_t n, int32_t d) {
    return (n + d - 1) / d;
  }

} // namespace

// Generic ABI used by FNGPULowerToRuntime.
//
extern "C" void __fngpu_launch_nd_f32(
				      int32_t kernelId,
				      int32_t rank,
				      int32_t blockX,
				      int32_t blockY,
				      int32_t blockZ,
				      float *a,
				      float *b,
				      float *c,
				      int32_t extentX,
				      int32_t extentY,
				      int32_t extentZ) {
  if (rank <= 0 || rank > 3) {
    std::fprintf(stderr, "FNGPU runtime error: unsupported rank %d\n", rank);
    std::abort();
  }

  if (extentX <= 0)
    return;

  if (rank >= 2 && extentY <= 0)
    return;

  if (rank >= 3 && extentZ <= 0)
    return;

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
                 "FNGPU runtime error: invalid block shape (%d,%d,%d)\n",
                 blockX, blockY, blockZ);
    std::abort();
  }

  CUfunction kernel = getKernelFunction(kernelId);

  // Current Triton pipeline uses num-warps=1 and emits .reqntid 32.
  constexpr int32_t cudaThreadsPerBlock = 32;

  int32_t gridX = ceilDiv(extentX, blockX);
  int32_t gridY = rank >= 2 ? ceilDiv(extentY, blockY) : 1;
  int32_t gridZ = rank >= 3 ? ceilDiv(extentZ, blockZ) : 1;

  // Current supported runtime data model:
  //   1-D f32 contiguous arrays.
  //
  // For now, use extentX for allocation size. When 2-D kernels are added,
  // change this to extentX * extentY, and for 3-D to X*Y*Z.
  size_t elementCount = static_cast<size_t>(extentX);
  if (rank >= 2)
    elementCount *= static_cast<size_t>(extentY);
  if (rank >= 3)
    elementCount *= static_cast<size_t>(extentZ);

  size_t bytes = elementCount * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNGPU_CUDA_CHECK(cuMemAlloc(&dA, bytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dB, bytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dC, bytes));

  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dA, a, bytes));
  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dB, b, bytes));

  CUdeviceptr extra0 = 0;
  CUdeviceptr extra1 = 0;

  if(rank == 1) {

    void *kernelArgs[] = {
      &dA,
      &dB,
      &dC,
      &extentX,
      &extra0,
      &extra1,
    };

    FNGPU_CUDA_CHECK(cuLaunchKernel(
				    kernel,
				    gridX, gridY, gridZ,
				    cudaThreadsPerBlock, 1, 1,
				    0,
				    nullptr,
				    kernelArgs,
				    nullptr));

  } else if (rank == 2) {

    void *kernelArgs[] = {
      &dA,
      &dB,
      &dC,
      &extentX,
      &extentY,
      &extra0,
      &extra1,
    };

    FNGPU_CUDA_CHECK(cuLaunchKernel(
				    kernel,
				    gridX, gridY, gridZ,
				    cudaThreadsPerBlock, 1, 1,
				    0,
				    nullptr,
				    kernelArgs,
				    nullptr));

  }

  FNGPU_CUDA_CHECK(cuCtxSynchronize());

  FNGPU_CUDA_CHECK(cuMemcpyDtoH(c, dC, bytes));

  FNGPU_CUDA_CHECK(cuMemFree(dA));
  FNGPU_CUDA_CHECK(cuMemFree(dB));
  FNGPU_CUDA_CHECK(cuMemFree(dC));
}



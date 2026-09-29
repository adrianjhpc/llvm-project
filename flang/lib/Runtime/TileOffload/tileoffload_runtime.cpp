#include "tileoffload_device.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

// For profiling
// Optional instrumentation; disabled builds do not require NVTX headers.
#if defined(TILEOFFLOAD_ENABLE_NVTX) && TILEOFFLOAD_ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#define TILEOFF_PROFILE_SCOPE(name) \
  nvtx3::scoped_range TileOffloadProfileRange { name }
#define TILEOFF_PROFILE_PUSH(name) nvtxRangePushA(name)
#define TILEOFF_PROFILE_POP() nvtxRangePop()
#else
#define TILEOFF_PROFILE_SCOPE(name) ((void)0)
#define TILEOFF_PROFILE_PUSH(name) ((void)0)
#define TILEOFF_PROFILE_POP() ((void)0)
#endif

#if defined(TILEOFFLOAD_RUNTIME_USE_HIP) && TILEOFFLOAD_RUNTIME_USE_HIP
#include <hip/hip_runtime_api.h>

// Keep the implementation below on one small driver-style API.  HIP's module
// API deliberately mirrors the CUDA driver API, so adapting it here avoids
// duplicating the cache, data-lifetime, launch, and reduction machinery.
using CUresult = hipError_t;
using CUdevice = hipDevice_t;
using CUcontext = hipCtx_t;
using CUstream = hipStream_t;
using CUevent = hipEvent_t;
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
using CUdeviceptr = std::uintptr_t;
using CUfunction_attribute = hipFunction_attribute;

static constexpr CUresult CUDA_SUCCESS = hipSuccess;
static constexpr CUresult CUDA_ERROR_NO_DEVICE = hipErrorNoDevice;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = hipErrorNotInitialized;
static constexpr CUresult CUDA_ERROR_ILLEGAL_ADDRESS = hipErrorIllegalAddress;
static constexpr unsigned CU_STREAM_DEFAULT = hipStreamDefault;
static constexpr unsigned CU_EVENT_DISABLE_TIMING = hipEventDisableTiming;
static constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK =
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
static constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES =
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
static constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_NUM_REGS =
    HIP_FUNC_ATTRIBUTE_NUM_REGS;
static constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_PTX_VERSION =
    HIP_FUNC_ATTRIBUTE_PTX_VERSION;
static constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_BINARY_VERSION =
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION;
static constexpr CUfunction_attribute
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES =
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;

static CUresult cuGetErrorName(CUresult error, const char **name) {
  *name = hipGetErrorName(error);
  return hipSuccess;
}
static CUresult cuGetErrorString(CUresult error, const char **description) {
  *description = hipGetErrorString(error);
  return hipSuccess;
}
static CUresult cuInit(unsigned flags) { return hipInit(flags); }
static CUresult cuDeviceGetCount(int *count) {
  return hipGetDeviceCount(count);
}
static CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  return hipDeviceGet(device, ordinal);
}
static CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
  return hipDevicePrimaryCtxRetain(context, device);
}
static CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
  return hipDevicePrimaryCtxRelease(device);
}
static CUresult cuCtxGetCurrent(CUcontext *context) {
  return hipCtxGetCurrent(context);
}
static CUresult cuCtxGetDevice(CUdevice *device) {
  return hipCtxGetDevice(device);
}
static CUresult cuCtxSetCurrent(CUcontext context) {
  return hipCtxSetCurrent(context);
}
static CUresult cuStreamCreate(CUstream *stream, unsigned flags) {
  return hipStreamCreateWithFlags(stream, flags);
}
static CUresult cuStreamDestroy(CUstream stream) {
  return hipStreamDestroy(stream);
}
static CUresult cuStreamSynchronize(CUstream stream) {
  return hipStreamSynchronize(stream);
}
static CUresult cuEventCreate(CUevent *event, unsigned flags) {
  return hipEventCreateWithFlags(event, flags);
}
static CUresult cuEventDestroy(CUevent event) { return hipEventDestroy(event); }
static CUresult cuEventRecord(CUevent event, CUstream stream) {
  return hipEventRecord(event, stream);
}
static CUresult cuEventSynchronize(CUevent event) {
  return hipEventSynchronize(event);
}
static CUresult cuModuleLoadDataEx(
    CUmodule *module, const void *image, unsigned, void *, void *) {
  return hipModuleLoadData(module, image);
}
static CUresult cuModuleUnload(CUmodule module) {
  return hipModuleUnload(module);
}
static CUresult cuModuleGetFunction(
    CUfunction *function, CUmodule module, const char *name) {
  return hipModuleGetFunction(function, module, name);
}
static CUresult cuFuncGetAttribute(
    int *value, CUfunction_attribute attribute, CUfunction function) {
  return hipFuncGetAttribute(value, attribute, function);
}
static CUresult cuFuncSetAttribute(CUfunction, CUfunction_attribute, int) {
  // AMD LDS does not use CUDA's per-function dynamic shared-memory opt-in.
  return hipSuccess;
}
static CUresult cuMemAlloc(CUdeviceptr *pointer, std::size_t bytes) {
  void *allocation = nullptr;
  CUresult result = hipMalloc(&allocation, bytes);
  if (result == hipSuccess)
    *pointer = reinterpret_cast<CUdeviceptr>(allocation);
  return result;
}
static CUresult cuMemFree(CUdeviceptr pointer) {
  return hipFree(reinterpret_cast<void *>(pointer));
}
static CUresult cuMemcpyHtoD(
    CUdeviceptr destination, const void *source, std::size_t bytes) {
  return hipMemcpy(reinterpret_cast<void *>(destination), source, bytes,
      hipMemcpyHostToDevice);
}
static CUresult cuMemcpyDtoH(
    void *destination, CUdeviceptr source, std::size_t bytes) {
  return hipMemcpy(destination, reinterpret_cast<const void *>(source), bytes,
      hipMemcpyDeviceToHost);
}
static CUresult cuMemcpyDtoDAsync(CUdeviceptr destination, CUdeviceptr source,
    std::size_t bytes, CUstream stream) {
  return hipMemcpyAsync(reinterpret_cast<void *>(destination),
      reinterpret_cast<const void *>(source), bytes, hipMemcpyDeviceToDevice,
      stream);
}
static CUresult cuLaunchKernel(CUfunction function, unsigned gridX,
    unsigned gridY, unsigned gridZ, unsigned blockX, unsigned blockY,
    unsigned blockZ, unsigned sharedBytes, CUstream stream, void **kernelParams,
    void **extra) {
  return hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedBytes, stream, kernelParams, extra);
}

static constexpr const char *TILEOFF_ACCELERATOR_NAME = "HIP";
#else
#include <cuda.h>
static constexpr const char *TILEOFF_ACCELERATOR_NAME = "CUDA";
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Lock order: lifetime -> registry -> (release registry) -> context.
// The shared lifetime lock prevents cleanup while an operation owns a state
// pointer. No GPU wait or transfer holds the registry lock on a warm context.
static std::recursive_mutex &TileOffloadGetRuntimeMutex() {
  static std::recursive_mutex mutex;
  return mutex;
}
static std::shared_mutex &TileOffloadGetLifetimeMutex() {
  static std::shared_mutex mutex;
  return mutex;
}
class TileOffloadOperationGuard;
#define TILEOFF_RUNTIME_GUARD() TileOffloadOperationGuard TileOffloadRuntimeLock
#define TILEOFF_REGISTRY_GUARD() \
  std::unique_lock<std::shared_mutex> TileOffloadLifetimeLock( \
      TileOffloadGetLifetimeMutex()); \
  std::lock_guard<std::recursive_mutex> TileOffloadRegistryLock( \
      TileOffloadGetRuntimeMutex())

static void TileOffloadCudaCheck(
    CUresult result, const char *expr, const char *file, int line) {
  if (result == CUDA_SUCCESS)
    return;

  const char *name = nullptr;
  const char *desc = nullptr;

  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);

  std::fprintf(stderr,
      "TileOffload %s driver error at %s:%d while executing %s: %s: %s\n",
      TILEOFF_ACCELERATOR_NAME, file, line, expr, name ? name : "<unknown>",
      desc ? desc : "<no description>");

  std::abort();
}

#define TILEOFF_CUDA_CHECK(expr) \
  do { \
    TileOffloadCudaCheck((expr), #expr, __FILE__, __LINE__); \
  } while (false)

static constexpr const char *TILEOFF_RUNTIME_BUILD_ID =
    "TILEOFF_RUNTIME_BUILD_ID_data_only_runtime_v13";

static std::size_t TileOffloadCheckedMul(
    std::size_t a, std::size_t b, const char *what) {
  if (a != 0 && b > static_cast<std::size_t>(-1) / a) {
    std::fprintf(
        stderr, "TileOffload error: size overflow while computing %s\n", what);
    std::abort();
  }

  return a * b;
}

static std::size_t TileOffloadCheckedAdd(
    std::size_t a, std::size_t b, const char *what) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    std::fprintf(
        stderr, "TileOffload error: size overflow while computing %s\n", what);
    std::abort();
  }
  return a + b;
}

// Properties belong to a loaded function in one context, not to a global
// kernel id. The context owns and invalidates the cache with its modules.
struct TileOffloadFunctionProperties {
  int maxThreadsPerBlock = 0;
  unsigned configuredDynamicSharedBytes = 0;
};
static TileOffloadFunctionProperties &TileOffloadGetFunctionProperties(CUfunction fn);

static void TileOffloadConfigureDynamicSharedMemory(
    CUfunction fn, int32_t kernelId, unsigned dynamicSharedBytes) {
  if (dynamicSharedBytes == 0)
    return;

  // 48 KiB is usually available without opt-in on many NVIDIA GPUs.
  // Above that, opt in if the device/function supports it.
  if (dynamicSharedBytes > 49152) {
    auto &properties = TileOffloadGetFunctionProperties(fn);
    if (dynamicSharedBytes <= properties.configuredDynamicSharedBytes)
      return;
    CUresult result =
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(dynamicSharedBytes));

    if (result != CUDA_SUCCESS) {
      const char *name = nullptr;
      const char *desc = nullptr;
      cuGetErrorName(result, &name);
      cuGetErrorString(result, &desc);

      std::fprintf(stderr,
          "TileOffload error: could not set dynamic shared memory size for kernel "
          "id %d to %u bytes: %s: %s\n",
          kernelId, dynamicSharedBytes, name ? name : "<unknown>",
          desc ? desc : "<no description>");
      std::abort();
    }
    properties.configuredDynamicSharedBytes = dynamicSharedBytes;
  }
}

static unsigned TileOffloadGetEnvUnsignedAllowZero(
    const char *name, unsigned fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;

  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);

  if (end == value || *end != '\0' ||
      parsed >
          static_cast<unsigned long>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "TileOffload error: invalid %s value '%s'\n", name, value);
    std::abort();
  }

  return static_cast<unsigned>(parsed);
}

static std::size_t TileOffloadCheckedBytes2D(
    int32_t dim0, int32_t dim1, std::size_t elemBytes, const char *what) {
  if (dim0 < 0 || dim1 < 0) {
    std::fprintf(stderr,
        "TileOffload error: negative dimension while computing %s: (%d,%d)\n", what,
        dim0, dim1);
    std::abort();
  }

  std::size_t elements = TileOffloadCheckedMul(
      static_cast<std::size_t>(dim0), static_cast<std::size_t>(dim1), what);

  return TileOffloadCheckedMul(elements, elemBytes, what);
}

static void TileOffloadValidateCudaBlockSize(
    CUfunction fn, int32_t kernelId, unsigned cudaBlockX) {
  auto &properties = TileOffloadGetFunctionProperties(fn);
  if (properties.maxThreadsPerBlock == 0)
    TILEOFF_CUDA_CHECK(cuFuncGetAttribute(&properties.maxThreadsPerBlock,
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
  int maxThreadsPerBlock = properties.maxThreadsPerBlock;

  if (cudaBlockX > static_cast<unsigned>(maxThreadsPerBlock)) {
    std::fprintf(stderr,
        "TileOffload error: kernel id %d requested CUDA block size %u, "
        "but function max_threads_per_block is %d\n",
        kernelId, cudaBlockX, maxThreadsPerBlock);
    std::abort();
  }
}

// Explicit API selection is per host thread and takes precedence over env
// settings.
static thread_local int TileOffloadSelectedDeviceOrdinal = -1;

static int TileOffloadGetDeviceOrdinal() {
  if (TileOffloadSelectedDeviceOrdinal >= 0)
    return TileOffloadSelectedDeviceOrdinal;
  const char *variable = "TILEOFF_DEVICE";
  const char *value = std::getenv(variable);
  if (!value || value[0] == '\0') {
#if defined(TILEOFFLOAD_RUNTIME_USE_HIP) && TILEOFFLOAD_RUNTIME_USE_HIP
    variable = "TILEOFF_HIP_DEVICE";
#else
    variable = "TILEOFF_CUDA_DEVICE";
#endif
    value = std::getenv(variable);
  }
  if (!value || value[0] == '\0')
    return 0;

  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr, "TileOffload error: invalid %s value '%s'\n", variable, value);
    std::abort();
  }

  return static_cast<int>(parsed);
}

// -------------------------------------------------------------------------- //
// Tiny dependency-free JSON helpers
// -------------------------------------------------------------------------- //
//
// These intentionally parse only the JSON shape emitted by the TileOffload compiler.
// This is not a general-purpose JSON parser.
//
// Expected generated object form:
//
// {
//   "id": 5,
//   "name": "tileoff_kernel_5",
//   "kind": "saxpy1d",
//   "rank": 1,
//   "tile": [128, 1, 1],
//   "num_warps": 1,
//   "threads_per_warp": 32,
//   "num_ctas": 1,
//   "num_stages": 3,
//   "cuda_threads_per_cta": 32,
//   ...
// }

static std::size_t jsonFindKey(const std::string &text, const char *key) {
  std::string quotedKey = "\"";
  quotedKey += key;
  quotedKey += "\"";
  return text.find(quotedKey);
}

static bool jsonFindInt(
    const std::string &text, const char *key, int32_t &out) {
  std::size_t keyPos = jsonFindKey(text, key);
  if (keyPos == std::string::npos)
    return false;

  std::size_t colon = text.find(':', keyPos);
  if (colon == std::string::npos)
    return false;

  const char *begin = text.c_str();
  const char *cursor = begin + colon + 1;
  const char *endOfString = begin + text.size();

  while (cursor < endOfString &&
      (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' ||
          *cursor == '\r')) {
    ++cursor;
  }

  errno = 0;
  char *end = nullptr;
  long long value = std::strtoll(cursor, &end, 10);
  if (end == cursor || errno == ERANGE ||
      value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max())
    return false;

  while (end < endOfString &&
      (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'))
    ++end;
  if (end < endOfString && *end != ',' && *end != '}' && *end != ']')
    return false;

  out = static_cast<int32_t>(value);
  return true;
}

static bool jsonFindBool(const std::string &text, const char *key, bool &out) {
  std::size_t keyPos = jsonFindKey(text, key);
  if (keyPos == std::string::npos)
    return false;

  std::size_t colon = text.find(':', keyPos);
  if (colon == std::string::npos)
    return false;

  std::size_t value = text.find_first_not_of(" \t\n\r", colon + 1);
  if (value == std::string::npos)
    return false;
  if (text.compare(value, 4, "true") == 0) {
    out = true;
    return true;
  }
  if (text.compare(value, 5, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

static bool jsonFindString(
    const std::string &text, const char *key, std::string &out) {
  std::size_t keyPos = jsonFindKey(text, key);
  if (keyPos == std::string::npos)
    return false;

  std::size_t colon = text.find(':', keyPos);
  if (colon == std::string::npos)
    return false;

  std::size_t quoteStart = text.find('"', colon + 1);
  if (quoteStart == std::string::npos)
    return false;

  std::size_t quoteEnd = text.find('"', quoteStart + 1);
  if (quoteEnd == std::string::npos)
    return false;

  out = text.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
  return true;
}

static const char *skipToNextIntegerLike(
    const char *cursor, const char *endOfString) {
  while (cursor < endOfString) {
    char c = *cursor;
    if ((c >= '0' && c <= '9') || c == '-')
      return cursor;
    ++cursor;
  }

  return cursor;
}

static bool jsonFindIntArray3(const std::string &text, const char *key,
    int32_t &x, int32_t &y, int32_t &z) {
  std::size_t keyPos = jsonFindKey(text, key);
  if (keyPos == std::string::npos)
    return false;

  std::size_t open = text.find('[', keyPos);
  if (open == std::string::npos)
    return false;

  const char *begin = text.c_str();
  const char *endOfString = begin + text.size();

  const char *cursor = begin + open + 1;
  char *end = nullptr;

  cursor = skipToNextIntegerLike(cursor, endOfString);
  if (cursor >= endOfString)
    return false;

  long v0 = std::strtol(cursor, &end, 10);
  if (end == cursor)
    return false;

  cursor = skipToNextIntegerLike(end, endOfString);
  if (cursor >= endOfString)
    return false;

  long v1 = std::strtol(cursor, &end, 10);
  if (end == cursor)
    return false;

  cursor = skipToNextIntegerLike(end, endOfString);
  if (cursor >= endOfString)
    return false;

  long v2 = std::strtol(cursor, &end, 10);
  if (end == cursor)
    return false;

  x = static_cast<int32_t>(v0);
  y = static_cast<int32_t>(v1);
  z = static_cast<int32_t>(v2);
  return true;
}

static bool jsonFindArrayText(
    const std::string &text, const char *key, std::string &out) {
  std::size_t keyPos = jsonFindKey(text, key);
  if (keyPos == std::string::npos)
    return false;

  std::size_t open = text.find('[', keyPos);
  if (open == std::string::npos)
    return false;

  bool inString = false;
  bool escaped = false;
  int depth = 0;

  for (std::size_t i = open; i < text.size(); ++i) {
    char c = text[i];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (c == '\\' && inString) {
      escaped = true;
      continue;
    }

    if (c == '"') {
      inString = !inString;
      continue;
    }

    if (inString)
      continue;

    if (c == '[') {
      ++depth;
      continue;
    }

    if (c == ']') {
      --depth;
      if (depth == 0) {
        out = text.substr(open + 1, i - open - 1);
        return true;
      }
    }
  }

  return false;
}

static void TileOffloadValidateContiguousDescriptor(const char *operationName,
    int64_t elementBytes, int32_t rank, int64_t extent0, int64_t extent1,
    int64_t extent2, int64_t stride0, int64_t stride1, int64_t stride2) {
  if (rank < 1 || rank > 3) {
    std::fprintf(stderr, "TileOffload error: %s received unsupported rank %d\n",
        operationName, rank);
    std::abort();
  }

  if (elementBytes <= 0) {
    std::fprintf(stderr, "TileOffload error: %s received invalid element size %lld\n",
        operationName, static_cast<long long>(elementBytes));
    std::abort();
  }

  if (extent0 < 0 || extent1 < 0 || extent2 < 0) {
    std::fprintf(
        stderr, "TileOffload error: %s received a negative extent\n", operationName);
    std::abort();
  }

  std::size_t expected0Size = static_cast<std::size_t>(elementBytes);
  std::size_t expected1Size = TileOffloadCheckedMul(expected0Size,
      static_cast<std::size_t>(extent0), "descriptor byte stride 1");
  std::size_t expected2Size = TileOffloadCheckedMul(expected1Size,
      static_cast<std::size_t>(extent1), "descriptor byte stride 2");

  if (expected2Size >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    std::fprintf(stderr, "TileOffload error: %s descriptor stride exceeds i64\n",
        operationName);
    std::abort();
  }

  int64_t expected0 = static_cast<int64_t>(expected0Size);
  int64_t expected1 = static_cast<int64_t>(expected1Size);
  int64_t expected2 = static_cast<int64_t>(expected2Size);

  bool contiguous = true;

  if (rank >= 1 && stride0 != expected0)
    contiguous = false;
  if (rank >= 2 && stride1 != expected1)
    contiguous = false;
  if (rank >= 3 && stride2 != expected2)
    contiguous = false;

  if (!contiguous) {
    std::fprintf(stderr,
        "TileOffload error: %s only supports contiguous assumed-shape arrays; "
        "got rank=%d elementBytes=%lld extents=(%lld,%lld,%lld) "
        "byte_strides=(%lld,%lld,%lld), expected "
        "byte_strides=(%lld,%lld,%lld)\n",
        operationName, rank, static_cast<long long>(elementBytes),
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), static_cast<long long>(stride0),
        static_cast<long long>(stride1), static_cast<long long>(stride2),
        static_cast<long long>(expected0), static_cast<long long>(expected1),
        static_cast<long long>(expected2));
    std::abort();
  }
}

static constexpr int32_t TILEOFF_SUPPORTED_SCHEMA_VERSION = 1;
static constexpr int32_t TILEOFF_PACK_TARGET_HOST = 0;
static constexpr int32_t TILEOFF_PACK_TARGET_DEVICE = 1;

struct TileOffloadPackEntry {
  int32_t kernelArgSlot = -1;
  int32_t target = TILEOFF_PACK_TARGET_HOST;
};

static std::vector<TileOffloadPackEntry> jsonParsePackEntries(
    const std::string &kernelObjectText) {
  std::vector<TileOffloadPackEntry> entries;

  std::string packArray;
  if (!jsonFindArrayText(kernelObjectText, "pack", packArray))
    return entries;

  std::size_t pos = 0;

  while (true) {
    std::size_t slotKey = packArray.find("\"kernel_arg_slot\"", pos);
    if (slotKey == std::string::npos)
      break;

    std::size_t objectEnd = packArray.find('}', slotKey);
    if (objectEnd == std::string::npos)
      objectEnd = packArray.size();

    std::string objectText = packArray.substr(slotKey, objectEnd - slotKey);

    TileOffloadPackEntry entry;

    if (!jsonFindInt(objectText, "kernel_arg_slot", entry.kernelArgSlot)) {
      pos = objectEnd;
      continue;
    }

    if (!jsonFindInt(objectText, "target", entry.target))
      entry.target = TILEOFF_PACK_TARGET_HOST;

    if (entry.target != TILEOFF_PACK_TARGET_HOST &&
        entry.target != TILEOFF_PACK_TARGET_DEVICE) {
      std::fprintf(stderr,
          "TileOffload warning: invalid pack target %d for slot %d; "
          "defaulting to host\n",
          entry.target, entry.kernelArgSlot);
      entry.target = TILEOFF_PACK_TARGET_HOST;
    }

    entries.push_back(entry);
    pos = objectEnd + 1;
  }

  return entries;
}

static std::size_t findEnclosingObjectStart(
    const std::string &json, std::size_t pos) {
  while (true) {
    if (json[pos] == '{')
      return pos;

    if (pos == 0)
      break;

    --pos;
  }

  return std::string::npos;
}

static std::size_t findJsonObjectEnd(
    const std::string &json, std::size_t objectStart) {
  bool inString = false;
  bool escaped = false;
  int depth = 0;

  for (std::size_t i = objectStart; i < json.size(); ++i) {
    char c = json[i];

    if (escaped) {
      escaped = false;
      continue;
    }

    if (c == '\\' && inString) {
      escaped = true;
      continue;
    }

    if (c == '"') {
      inString = !inString;
      continue;
    }

    if (inString)
      continue;

    if (c == '{') {
      ++depth;
      continue;
    }

    if (c == '}') {
      --depth;
      if (depth == 0)
        return i + 1;
    }
  }

  return std::string::npos;
}

struct TileOffloadHiddenTritonArgs {
  // Triton/NVVM-generated PTX currently appends two hidden pointer parameters
  // after the explicit kernel parameters. For the kernels TileOffload currently
  // emits, these are not used, so null device pointers are sufficient.
  //
  // Example PTX:
  //
  //   .param .u64 ptr param_0  // explicit a
  //   .param .u64 ptr param_1  // explicit b
  //   .param .u64 ptr param_2  // explicit c
  //   .param .u32     param_3  // explicit n
  //   .param .u64 ptr param_4  // hidden
  //   .param .u64 ptr param_5  // hidden
  CUdeviceptr hidden0 = 0;
  CUdeviceptr hidden1 = 0;
};

enum class TileOffloadKernelParameterRole {
  Read,
  Write,
  ReadWrite,
  Partials,
  Scalar,
  ExtentX,
  ExtentY,
  ExtentZ,
  LoopLowerX,
  LoopLowerY,
  LoopLowerZ,
  ArrayLowerBound,
  ArrayStride,
  Unknown
};

struct TileOffloadKernelParameterDesc {
  int32_t slot = -1;
  TileOffloadKernelParameterRole role = TileOffloadKernelParameterRole::Unknown;
  std::string type;
  int32_t arrayIndex = -1;
  int32_t scalarIndex = -1;
  int32_t dimension = -1;
};

static TileOffloadKernelParameterRole TileOffloadParseParameterRole(
    const std::string &role) {
  if (role == "read")
    return TileOffloadKernelParameterRole::Read;
  if (role == "write")
    return TileOffloadKernelParameterRole::Write;
  if (role == "read_write")
    return TileOffloadKernelParameterRole::ReadWrite;
  if (role == "partials")
    return TileOffloadKernelParameterRole::Partials;
  if (role == "scalar")
    return TileOffloadKernelParameterRole::Scalar;
  if (role == "extent_x")
    return TileOffloadKernelParameterRole::ExtentX;
  if (role == "extent_y")
    return TileOffloadKernelParameterRole::ExtentY;
  if (role == "extent_k" || role == "extent_z")
    return TileOffloadKernelParameterRole::ExtentZ;
  if (role == "loop_lower_x")
    return TileOffloadKernelParameterRole::LoopLowerX;
  if (role == "loop_lower_y")
    return TileOffloadKernelParameterRole::LoopLowerY;
  if (role == "loop_lower_z")
    return TileOffloadKernelParameterRole::LoopLowerZ;
  if (role == "array_lower_bound")
    return TileOffloadKernelParameterRole::ArrayLowerBound;
  if (role == "array_stride")
    return TileOffloadKernelParameterRole::ArrayStride;
  return TileOffloadKernelParameterRole::Unknown;
}

static std::vector<TileOffloadKernelParameterDesc> jsonParseParameterEntries(
    const std::string &kernelObjectText) {
  std::vector<TileOffloadKernelParameterDesc> parameters;
  std::string paramsArray;
  if (!jsonFindArrayText(kernelObjectText, "params", paramsArray))
    return parameters;

  std::size_t pos = 0;
  int32_t nextImplicitScalarIndex = 0;
  while (true) {
    std::size_t slotKey = paramsArray.find("\"slot\"", pos);
    if (slotKey == std::string::npos)
      break;
    std::size_t objectStart = paramsArray.rfind('{', slotKey);
    if (objectStart == std::string::npos)
      break;
    std::size_t objectEnd = findJsonObjectEnd(paramsArray, objectStart);
    if (objectEnd == std::string::npos)
      break;

    std::string objectText =
        paramsArray.substr(objectStart, objectEnd - objectStart);
    TileOffloadKernelParameterDesc parameter;
    std::string role;
    if (!jsonFindInt(objectText, "slot", parameter.slot) ||
        !jsonFindString(objectText, "role", role) ||
        !jsonFindString(objectText, "type", parameter.type)) {
      pos = objectEnd;
      continue;
    }
    parameter.role = TileOffloadParseParameterRole(role);
    jsonFindInt(objectText, "array_index", parameter.arrayIndex);
    jsonFindInt(objectText, "scalar_index", parameter.scalarIndex);
    jsonFindInt(objectText, "dimension", parameter.dimension);
    // Stencil metadata emitted before the Stage 1 migration ordered scalars
    // but did not spell out scalar_index. Preserve compatibility with those
    // launch-ABI-v2 objects while making new metadata explicit.
    if (parameter.role == TileOffloadKernelParameterRole::Scalar) {
      if (parameter.scalarIndex < 0)
        parameter.scalarIndex = nextImplicitScalarIndex;
      nextImplicitScalarIndex =
          std::max(nextImplicitScalarIndex, parameter.scalarIndex + 1);
    }
    parameters.push_back(std::move(parameter));
    pos = objectEnd;
  }

  std::sort(parameters.begin(), parameters.end(),
      [](const TileOffloadKernelParameterDesc &lhs,
          const TileOffloadKernelParameterDesc &rhs) { return lhs.slot < rhs.slot; });
  return parameters;
}

struct TileOffloadKernelDesc {
  int32_t id = -1;
  std::string name;
  std::string kind = "binary";
  bool isReduction = false;
  bool isMatmul = false;

  int32_t rank = 1;

  int32_t loopStep[3] = {1, 1, 1};
  int32_t loopStepScalarIndex[3] = {-1, -1, -1};
  int32_t tileX = 1024;
  int32_t tileY = 1;
  int32_t tileZ = 1;

  int32_t numWarps = 1;
  int32_t threadsPerWarp = 32;
  int32_t numCTAs = 1;
  int32_t numStages = 3;

  int32_t cudaThreadsPerCTA = 32;

  // Number of hidden pointer parameters appended by Triton/NVVM PTX.
  int32_t tritonHiddenPtrArgs = 2;

  std::string backend = "triton";
  std::string acceleratorTarget = "cuda";
  std::string deviceImageKind = "ptx";

  // Synthetic kernel used to recursively reduce a partials buffer. A negative
  // value denotes older metadata that requires the host-side fallback.
  int32_t reductionStageId = -1;

  int32_t launchAbiVersion = 1;
  int32_t arrayCount = 0;
  int32_t scalarCount = 0;
  int32_t outputCount = 0;
  bool copyBackWrites = true;

  enum class ReductionOperator { Add, Multiply, Min, Max };
  ReductionOperator reductionOp = ReductionOperator::Add;

  // PACK metadata from JSON.
  std::vector<TileOffloadPackEntry> pack;

  // Ordered, source-visible device parameters used by launch ABI v2.
  std::vector<TileOffloadKernelParameterDesc> parameters;

  // Derived only from validated, immutable v2 metadata. Device residency and
  // actual array/scalar bindings are deliberately not cached here.
  std::vector<std::optional<int32_t>> explicitArrayTargets;
  std::vector<std::size_t> scalarParameterBytes;

  int32_t ptxIndex = 0;
  std::string ptxFile;
};

static std::size_t TileOffloadScalarParameterBytes(const std::string &type) {
  if (type == "i8")
    return sizeof(int8_t);
  if (type == "i16")
    return sizeof(int16_t);
  if (type == "i32" || type == "f32")
    return sizeof(int32_t);
  if (type == "i64" || type == "f64")
    return sizeof(int64_t);
  return 0;
}

static bool TileOffloadIsPointerParameterType(const std::string &type) {
  return type.size() > 5 && type.compare(0, 4, "ptr<") == 0 &&
      type.back() == '>';
}

// Build once after metadata validation, before publishing the descriptor.
static void TileOffloadPrepareVariadicMetadata(TileOffloadKernelDesc &desc) {
  desc.isReduction = desc.kind == "reduction_sum1d" ||
      desc.kind == "reduction_dot1d" || desc.kind == "reduction_product1d" ||
      desc.kind == "reduction_min1d" || desc.kind == "reduction_max1d" ||
      desc.kind == "reduction_multi2d";
  desc.isMatmul = desc.kind == "matmul2d";
  desc.explicitArrayTargets.assign(desc.arrayCount, std::nullopt);
  desc.scalarParameterBytes.assign(desc.parameters.size(), 0);
  for (const TileOffloadKernelParameterDesc &parameter : desc.parameters) {
    if (parameter.role == TileOffloadKernelParameterRole::Scalar)
      desc.scalarParameterBytes[parameter.slot] =
          TileOffloadScalarParameterBytes(parameter.type);
    if (parameter.role != TileOffloadKernelParameterRole::Read &&
        parameter.role != TileOffloadKernelParameterRole::Write &&
        parameter.role != TileOffloadKernelParameterRole::ReadWrite)
      continue;
    auto &target = desc.explicitArrayTargets[parameter.arrayIndex];
    for (const TileOffloadPackEntry &entry : desc.pack) {
      if (entry.kernelArgSlot != parameter.slot)
        continue;
      if (target && *target != entry.target) {
        std::fprintf(stderr,
            "TileOffload error: conflicting PACK targets for v2 array %d in "
            "kernel id %d\n",
            parameter.arrayIndex, desc.id);
        std::abort();
      }
      target = entry.target;
      // Match the existing first-entry lookup for each parameter slot.
      break;
    }
  }
}

static void TileOffloadValidateVariadicKernelMetadata(const TileOffloadKernelDesc &desc) {
  for (int dim = 0; dim < 3; ++dim) {
    int slot = desc.loopStepScalarIndex[dim];
    bool valid = slot == -1
        ? desc.loopStep[dim] != 0
        : (slot >= 0 && slot < desc.scalarCount && desc.loopStep[dim] == 0 &&
              dim < (desc.kind == "matmul2d" ? 3 : desc.rank));
    if (slot >= 0) {
      valid &= std::any_of(desc.parameters.begin(), desc.parameters.end(),
          [&](const TileOffloadKernelParameterDesc &p) {
            return p.role == TileOffloadKernelParameterRole::Scalar &&
                p.scalarIndex == slot && p.type == "i32";
          });
    }
    if (!valid) {
      std::fprintf(
          stderr, "TileOffload error: invalid loop step scalar binding metadata\n");
      std::abort();
    }
  }
  if (desc.rank < 1 || desc.rank > 2 || desc.arrayCount <= 0 ||
      desc.scalarCount < 0 || desc.outputCount <= 0 ||
      desc.parameters.empty()) {
    std::fprintf(stderr,
        "TileOffload error: invalid v2 ABI metadata for kernel id %d\n", desc.id);
    std::abort();
  }

  std::vector<bool> arraysReferenced(
      static_cast<std::size_t>(desc.arrayCount), false);
  std::vector<bool> scalarsReferenced(
      static_cast<std::size_t>(desc.scalarCount), false);
  int32_t extentCounts[3] = {0, 0, 0};
  int32_t partialsCount = 0;
  bool isReduction = desc.kind == "reduction_sum1d" ||
      desc.kind == "reduction_dot1d" || desc.kind == "reduction_product1d" ||
      desc.kind == "reduction_min1d" || desc.kind == "reduction_max1d" ||
      desc.kind == "reduction_multi2d";
  bool isMultiReduction = desc.kind == "reduction_multi2d";
  bool isMatmul = desc.kind == "matmul2d";

  if (isReduction && !isMultiReduction && desc.outputCount != 1) {
    std::fprintf(stderr,
        "TileOffload error: invalid v2 reduction output count for kernel id %d\n",
        desc.id);
    std::abort();
  }

  for (std::size_t index = 0; index < desc.parameters.size(); ++index) {
    const TileOffloadKernelParameterDesc &parameter = desc.parameters[index];
    if (parameter.slot != static_cast<int32_t>(index)) {
      std::fprintf(stderr,
          "TileOffload error: non-contiguous v2 parameter slots for kernel id %d\n",
          desc.id);
      std::abort();
    }

    auto validateArray = [&](bool isPointerParameter = true) {
      if (parameter.arrayIndex < 0 || parameter.arrayIndex >= desc.arrayCount) {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 array index for kernel id %d slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      if (isPointerParameter)
        arraysReferenced[static_cast<std::size_t>(parameter.arrayIndex)] = true;
    };

    switch (parameter.role) {
    case TileOffloadKernelParameterRole::Read:
    case TileOffloadKernelParameterRole::Write:
    case TileOffloadKernelParameterRole::ReadWrite:
      validateArray();
      if (!TileOffloadIsPointerParameterType(parameter.type)) {
        std::fprintf(stderr,
            "TileOffload error: v2 array parameter is not a pointer for kernel id "
            "%d slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    case TileOffloadKernelParameterRole::Scalar:
      if (parameter.scalarIndex < 0 ||
          parameter.scalarIndex >= desc.scalarCount ||
          TileOffloadScalarParameterBytes(parameter.type) == 0) {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 scalar metadata for kernel id %d slot "
            "%d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      scalarsReferenced[static_cast<std::size_t>(parameter.scalarIndex)] = true;
      break;
    case TileOffloadKernelParameterRole::ExtentX:
    case TileOffloadKernelParameterRole::ExtentY:
    case TileOffloadKernelParameterRole::ExtentZ: {
      unsigned dim = parameter.role == TileOffloadKernelParameterRole::ExtentX ? 0
          : parameter.role == TileOffloadKernelParameterRole::ExtentY          ? 1
                                                                         : 2;
      bool validDimension =
          dim < static_cast<unsigned>(desc.rank) || (isMatmul && dim == 2);
      if (!validDimension || parameter.type != "i32") {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 extent metadata for kernel id %d slot "
            "%d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      ++extentCounts[dim];
      break;
    }
    case TileOffloadKernelParameterRole::LoopLowerX:
    case TileOffloadKernelParameterRole::LoopLowerY:
    case TileOffloadKernelParameterRole::LoopLowerZ: {
      unsigned dim = parameter.role == TileOffloadKernelParameterRole::LoopLowerX ? 0
          : parameter.role == TileOffloadKernelParameterRole::LoopLowerY          ? 1
                                                                            : 2;
      if ((dim >= static_cast<unsigned>(desc.rank) &&
              !(desc.kind == "matmul2d" && dim == 2)) ||
          parameter.type != "i32") {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 loop-lower metadata for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    }
    case TileOffloadKernelParameterRole::ArrayLowerBound:
    case TileOffloadKernelParameterRole::ArrayStride:
      validateArray(false);
      if (parameter.dimension < 0 || parameter.dimension >= desc.rank ||
          parameter.type != "i32") {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 array-layout metadata for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    case TileOffloadKernelParameterRole::Partials:
      if (!isReduction || !TileOffloadIsPointerParameterType(parameter.type)) {
        std::fprintf(stderr,
            "TileOffload error: invalid v2 partials parameter for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      ++partialsCount;
      break;
    case TileOffloadKernelParameterRole::Unknown:
      std::fprintf(stderr,
          "TileOffload error: unsupported v2 parameter role for kernel id %d slot "
          "%d\n",
          desc.id, parameter.slot);
      std::abort();
    }
  }

  if (extentCounts[0] != 1 || (desc.rank >= 2 && extentCounts[1] != 1) ||
      (isMatmul ? extentCounts[2] != 1 : extentCounts[2] != 0) ||
      (isReduction ? partialsCount != 1 : partialsCount != 0) ||
      std::find(arraysReferenced.begin(), arraysReferenced.end(), false) !=
          arraysReferenced.end() ||
      std::find(scalarsReferenced.begin(), scalarsReferenced.end(), false) !=
          scalarsReferenced.end()) {
    std::fprintf(stderr,
        "TileOffload error: incomplete v2 parameter metadata for kernel id %d\n",
        desc.id);
    std::abort();
  }
}

struct TileOffloadDeviceAllocation {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
  std::size_t dataRegionReferences = 0;
};

struct TileOffloadDataRegionFrame {
  std::vector<void *> allocations;
};

struct TileOffloadReductionBufferStats {
  // Number of cuMemAlloc calls. Growth allocations are included here and are
  // also counted separately below.
  uint64_t allocations = 0;
  uint64_t growths = 0;
  uint64_t reuses = 0;
};

struct TileOffloadReductionWorkspace {
  CUdevice device = 0;
  CUcontext context = nullptr;

  // The primary kernel writes one partial per Triton program to this buffer.
  TileOffloadDeviceAllocation partials;
  TileOffloadReductionBufferStats partialStats;

  // Hierarchical stages ping-pong between partials and this buffer.
  TileOffloadDeviceAllocation scratch;
  TileOffloadReductionBufferStats scratchStats;

  // Packed final results let multi-reductions share one host transfer.
  TileOffloadDeviceAllocation results;
  TileOffloadReductionBufferStats resultStats;
  uint64_t primaryLaunches = 0;
  uint64_t stageLaunches = 0;
};

struct TileOffloadContextState {
  std::shared_ptr<std::recursive_mutex> mutex =
      std::make_shared<std::recursive_mutex>();
  CUdevice device = 0;
  CUcontext context = nullptr;
  bool retainedPrimaryContext = false;
  CUstream stream = nullptr;
  CUevent completionEvent = nullptr;
  std::vector<CUmodule> modules;
  std::unordered_map<int32_t, CUfunction> functionCache;
  std::unordered_map<CUfunction, TileOffloadFunctionProperties> functionProperties;
  bool pendingResidentLaunches = false;
  std::unordered_map<void *, TileOffloadDeviceAllocation> deviceCache;
  std::vector<TileOffloadDataRegionFrame> dataRegions;
  TileOffloadReductionWorkspace reductionWorkspace;
};

struct TileOffloadKernelRegistry {
  bool initialized = false;

  std::unordered_map<int32_t, TileOffloadKernelDesc> kernels;
  std::vector<std::string> ptxTexts;
  std::vector<int32_t> embeddedImageKinds;

  // CUDA modules, functions, allocations, streams and reduction buffers are
  // all context-owned. Never reuse any of them in a different context, even
  // when two contexts select the same CUDA device.
  std::unordered_map<CUcontext, TileOffloadContextState> contexts;
  std::unordered_map<int, CUcontext> primaryContexts;
};

struct TileOffloadEmbeddedKernelBundle {
  // Keep these integer values in sync with the generated C bundle ABI.
  enum ImageKind : int32_t { PTX = 1, Cubin = 2, HSACO = 3 };

  std::vector<const void *> imageData;
  std::vector<std::size_t> imageSize;
  std::vector<int32_t> imageKind;
  const char *jsonData = nullptr;
  std::size_t jsonSize = 0;
};

// The generated bundle is registered from a constructor in another
// translation unit.  A function-local static prevents that constructor from
// writing into namespace-scope vectors before their constructors have run.
static std::vector<TileOffloadEmbeddedKernelBundle> &TileOffloadGetEmbeddedKernelBundles() {
  static std::vector<TileOffloadEmbeddedKernelBundle> bundles;
  return bundles;
}

static TileOffloadKernelRegistry TileOffloadRegistry;

static thread_local TileOffloadContextState *TileOffloadActiveState = nullptr;
static thread_local CUcontext TileOffloadActiveContext = nullptr;
static TileOffloadContextState &TileOffloadActiveContextState() {
  if (!TileOffloadActiveState) {
    std::fprintf(stderr, "TileOffload error: no active CUDA context state\n");
    std::abort();
  }
  return *TileOffloadActiveState;
}

static bool TileOffloadReadEnvFlag(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
class TileOffloadOperationGuard;
static thread_local TileOffloadOperationGuard *TileOffloadOperation = nullptr;
class TileOffloadOperationGuard {
public:
  TileOffloadOperationGuard() : parent(TileOffloadOperation) {
    if (!parent) {
      lifetime = std::shared_lock<std::shared_mutex>(TileOffloadGetLifetimeMutex());
      registry = std::unique_lock<std::recursive_mutex>(TileOffloadGetRuntimeMutex());
      debug = TileOffloadReadEnvFlag("TILEOFF_DEBUG");
      asyncResident = TileOffloadReadEnvFlag("TILEOFF_ASYNC_RESIDENT");
      reductionStats = TileOffloadReadEnvFlag("TILEOFF_REDUCTION_STATS");
    } else {
      debug = parent->debug;
      asyncResident = parent->asyncResident;
      reductionStats = parent->reductionStats;
    }
    TileOffloadOperation = this;
  }
  ~TileOffloadOperationGuard() {
    TileOffloadOperation = parent;
    if (!parent) {
      TileOffloadActiveState = nullptr;
      TileOffloadActiveContext = nullptr;
    }
  }
  TileOffloadOperationGuard(const TileOffloadOperationGuard &) = delete;
  TileOffloadOperationGuard &operator=(const TileOffloadOperationGuard &) = delete;
  void select(TileOffloadContextState &state) {
    if (parent) {
      parent->select(state);
      return;
    }
    if (!context.owns_lock()) {
      // Release before waiting: another context must remain able to launch.
      registry.unlock();
      context = std::unique_lock<std::recursive_mutex>(*state.mutex);
    }
    TileOffloadActiveState = &state;
    TileOffloadActiveContext = state.context;
  }
  bool debug = false, asyncResident = false, reductionStats = false;

private:
  TileOffloadOperationGuard *parent;
  std::shared_lock<std::shared_mutex> lifetime;
  std::unique_lock<std::recursive_mutex> registry;
  std::unique_lock<std::recursive_mutex> context;
};

struct TileOffloadDeviceArg {
  CUdeviceptr ptr = 0;
  bool cached = false;
  int32_t target = TILEOFF_PACK_TARGET_HOST;
  int32_t slot = -1;
};

static TileOffloadFunctionProperties &TileOffloadGetFunctionProperties(CUfunction fn) {
  return TileOffloadActiveContextState().functionProperties[fn];
}

static bool TileOffloadEnvFlagEnabled(const char *name) {
  if (TileOffloadOperation) {
    if (std::strcmp(name, "TILEOFF_DEBUG") == 0)
      return TileOffloadOperation->debug;
    if (std::strcmp(name, "TILEOFF_ASYNC_RESIDENT") == 0)
      return TileOffloadOperation->asyncResident;
    if (std::strcmp(name, "TILEOFF_REDUCTION_STATS") == 0)
      return TileOffloadOperation->reductionStats;
  }
  return TileOffloadReadEnvFlag(name);
}

static bool TileOffloadDebugEnabled() {
  return TileOffloadOperation ? TileOffloadOperation->debug
                        : TileOffloadReadEnvFlag("TILEOFF_DEBUG");
}

static bool TileOffloadReductionStatsEnabled() {
  return TileOffloadEnvFlagEnabled("TILEOFF_REDUCTION_STATS");
}

// Reuse context selection only inside a guarded runtime call. Never retain
// this decision across calls: the caller may change its context or device.
class TileOffloadCurrentContextGuard;
static thread_local TileOffloadCurrentContextGuard *TileOffloadContextScope = nullptr;

class TileOffloadCurrentContextGuard {
public:
  TileOffloadCurrentContextGuard() : parent(TileOffloadContextScope) {
    if (parent && parent->selectedContext) {
      previousContext = selectedContext = parent->selectedContext;
    } else {
      CUresult result = cuCtxGetCurrent(&previousContext);
      if (result == CUDA_ERROR_NOT_INITIALIZED)
        previousContext = nullptr;
      else
        TILEOFF_CUDA_CHECK(result);
    }
    TileOffloadContextScope = this;
  }

  TileOffloadCurrentContextGuard(const TileOffloadCurrentContextGuard &) = delete;
  TileOffloadCurrentContextGuard &operator=(
      const TileOffloadCurrentContextGuard &) = delete;

  ~TileOffloadCurrentContextGuard() {
    TileOffloadContextScope = parent;
    if (selectedContext && selectedContext == previousContext)
      return;
    CUresult result = cuCtxSetCurrent(previousContext);
    if (result == CUDA_ERROR_NOT_INITIALIZED)
      return;
    if (result != CUDA_SUCCESS && TileOffloadDebugEnabled())
      std::fprintf(stderr,
          "TileOffload warning: failed to restore the caller's CUDA context\n");
  }

  bool isSelected() const { return selectedContext != nullptr; }
  CUcontext callerContext() const { return previousContext; }
  void select(CUcontext context) { selectedContext = context; }

private:
  TileOffloadCurrentContextGuard *parent;
  CUcontext previousContext{nullptr};
  CUcontext selectedContext{nullptr};
};

static TileOffloadReductionWorkspace TileOffloadAggregateReductionWorkspaceStats() {
  TileOffloadReductionWorkspace total;
  for (const auto &entry : TileOffloadRegistry.contexts) {
    const TileOffloadReductionWorkspace &workspace = entry.second.reductionWorkspace;
    total.primaryLaunches += workspace.primaryLaunches;
    total.stageLaunches += workspace.stageLaunches;
    total.partialStats.allocations += workspace.partialStats.allocations;
    total.partialStats.growths += workspace.partialStats.growths;
    total.partialStats.reuses += workspace.partialStats.reuses;
    total.partials.bytes += workspace.partials.bytes;
    total.scratchStats.allocations += workspace.scratchStats.allocations;
    total.scratchStats.growths += workspace.scratchStats.growths;
    total.scratchStats.reuses += workspace.scratchStats.reuses;
    total.scratch.bytes += workspace.scratch.bytes;
  }
  return total;
}

static void TileOffloadPrintReductionWorkspaceStats() {
  TileOffloadReductionWorkspace workspace = TileOffloadAggregateReductionWorkspaceStats();
  std::fprintf(stderr,
      "TileOffload reduction workspace: primary_launches=%llu "
      "stage_launches=%llu "
      "contexts=%zu "
      "partials={allocations=%llu,growths=%llu,reuses=%llu,capacity_bytes=%zu} "
      "scratch={allocations=%llu,growths=%llu,reuses=%llu,capacity_bytes=%zu}"
      "\n",
      static_cast<unsigned long long>(workspace.primaryLaunches),
      static_cast<unsigned long long>(workspace.stageLaunches),
      TileOffloadRegistry.contexts.size(),
      static_cast<unsigned long long>(workspace.partialStats.allocations),
      static_cast<unsigned long long>(workspace.partialStats.growths),
      static_cast<unsigned long long>(workspace.partialStats.reuses),
      workspace.partials.bytes,
      static_cast<unsigned long long>(workspace.scratchStats.allocations),
      static_cast<unsigned long long>(workspace.scratchStats.growths),
      static_cast<unsigned long long>(workspace.scratchStats.reuses),
      workspace.scratch.bytes);
}

static const char *TileOffloadEmbeddedImageKindName(int32_t kind) {
  switch (kind) {
  case TileOffloadEmbeddedKernelBundle::PTX:
    return "ptx";
  case TileOffloadEmbeddedKernelBundle::Cubin:
    return "cubin";
  case TileOffloadEmbeddedKernelBundle::HSACO:
    return "hsaco";
  default:
    return "unknown";
  }
}

static bool TileOffloadHasEmbeddedBundles() {
  const auto &bundles = TileOffloadGetEmbeddedKernelBundles();
  if (bundles.empty())
    return false;
  for (const TileOffloadEmbeddedKernelBundle &bundle : bundles) {
    if (!bundle.jsonData || bundle.jsonSize == 0 || bundle.imageData.empty() ||
        bundle.imageData.size() != bundle.imageSize.size() ||
        bundle.imageData.size() != bundle.imageKind.size())
      return false;
    for (std::size_t i = 0; i < bundle.imageData.size(); ++i) {
      if (!bundle.imageData[i] || bundle.imageSize[i] == 0 ||
          (bundle.imageKind[i] != TileOffloadEmbeddedKernelBundle::PTX &&
              bundle.imageKind[i] != TileOffloadEmbeddedKernelBundle::Cubin &&
              bundle.imageKind[i] != TileOffloadEmbeddedKernelBundle::HSACO))
        return false;
    }
  }
  return true;
}

static std::vector<std::string> TileOffloadGetImagesFromEmbeddedBundles() {
  std::vector<std::string> result;

  if (!TileOffloadHasEmbeddedBundles())
    return result;

  const auto &bundles = TileOffloadGetEmbeddedKernelBundles();
  for (std::size_t bundleIndex = 0; bundleIndex < bundles.size();
      ++bundleIndex) {
    const TileOffloadEmbeddedKernelBundle &bundle = bundles[bundleIndex];
    for (std::size_t i = 0; i < bundle.imageData.size(); ++i) {
      const char *bytes = static_cast<const char *>(bundle.imageData[i]);
      result.emplace_back(bytes, bundle.imageSize[i]);
      if (TileOffloadDebugEnabled()) {
        std::fprintf(stderr,
            "TileOffload: loading embedded %s bundle=%zu entry=%zu bytes=%zu\n",
            TileOffloadEmbeddedImageKindName(bundle.imageKind[i]), bundleIndex, i,
            bundle.imageSize[i]);
      }
    }
  }
  return result;
}

static std::vector<int32_t> TileOffloadGetImageKindsFromEmbeddedBundles() {
  std::vector<int32_t> result;
  if (!TileOffloadHasEmbeddedBundles())
    return result;
  for (const TileOffloadEmbeddedKernelBundle &bundle :
      TileOffloadGetEmbeddedKernelBundles())
    result.insert(
        result.end(), bundle.imageKind.begin(), bundle.imageKind.end());
  return result;
}

static std::string TileOffloadReadTextFile(const char *path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    std::fprintf(stderr, "TileOffload error: could not open file '%s'\n", path);
    std::abort();
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

static bool TileOffloadTextFileExists(const char *path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  return static_cast<bool>(file);
}

static std::vector<std::string> TileOffloadGetPtxTextsFromDirectory(
    const std::unordered_map<int32_t, TileOffloadKernelDesc> &kernels) {
  std::vector<std::string> result;

  const char *dir = std::getenv("TILEOFF_PTX_DIR");
  if (!dir || dir[0] == '\0')
    return result;

  int32_t maxIndex = -1;
  for (const auto &entry : kernels)
    if (entry.second.ptxIndex > maxIndex)
      maxIndex = entry.second.ptxIndex;

  if (maxIndex < 0)
    return result;

  result.resize(static_cast<std::size_t>(maxIndex + 1));

  for (const auto &entry : kernels) {
    const TileOffloadKernelDesc &desc = entry.second;

    std::string path = std::string(dir) + "/" + desc.ptxFile;

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr, "TileOffload: loading PTX for kernel id %d from '%s'\n",
          desc.id, path.c_str());
    }

    result[static_cast<std::size_t>(desc.ptxIndex)] =
        TileOffloadReadTextFile(path.c_str());
  }

  return result;
}

static std::vector<std::string> TileOffloadGetPtxTexts(
    const std::unordered_map<int32_t, TileOffloadKernelDesc> &kernels) {
  const char *singlePtxPath = std::getenv("TILEOFF_PTX");

  if (singlePtxPath && singlePtxPath[0] != '\0') {
    if (TileOffloadDebugEnabled())
      std::fprintf(
          stderr, "TileOffload: loading single PTX from '%s'\n", singlePtxPath);

    return {TileOffloadReadTextFile(singlePtxPath)};
  }

  if (const char *dir = std::getenv("TILEOFF_PTX_DIR")) {
    if (dir[0] != '\0')
      return TileOffloadGetPtxTextsFromDirectory(kernels);
  }

  if (TileOffloadHasEmbeddedBundles())
    return TileOffloadGetImagesFromEmbeddedBundles();

  // Backwards-compatible fallback.
  return {TileOffloadReadTextFile("tileoff_kernels.ptx")};
}

static std::string TileOffloadGetJsonText() {
  const char *jsonPath = std::getenv("TILEOFF_KERNELS_JSON");

  if (jsonPath && jsonPath[0] != '\0') {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: loading JSON from '%s'\n", jsonPath);

    return TileOffloadReadTextFile(jsonPath);
  }

  if (TileOffloadHasEmbeddedBundles() &&
      TileOffloadGetEmbeddedKernelBundles().size() == 1) {
    const TileOffloadEmbeddedKernelBundle &bundle =
        TileOffloadGetEmbeddedKernelBundles().front();
    if (TileOffloadDebugEnabled()) {
      std::fprintf(
          stderr, "TileOffload: loading embedded JSON, bytes=%zu\n", bundle.jsonSize);
    }

    return std::string(bundle.jsonData, bundle.jsonSize);
  }

  const char *fallback = "tileoff_kernels.json";

  if (TileOffloadDebugEnabled())
    std::fprintf(stderr, "TileOffload: loading JSON from '%s'\n", fallback);

  return TileOffloadReadTextFile(fallback);
}

static std::unordered_map<int32_t, TileOffloadKernelDesc>
TileOffloadParseKernelDescsFromJson(const std::string &json) {
  std::unordered_map<int32_t, TileOffloadKernelDesc> result;

  std::size_t pos = 0;

  while (true) {
    std::size_t idKey = json.find("\"id\"", pos);
    if (idKey == std::string::npos)
      break;

    std::size_t objectStart = findEnclosingObjectStart(json, idKey);
    if (objectStart == std::string::npos) {
      pos = idKey + 4;
      continue;
    }

    std::size_t objectEnd = findJsonObjectEnd(json, objectStart);
    if (objectEnd == std::string::npos) {
      std::fprintf(stderr, "TileOffload error: malformed kernel JSON object\n");
      std::abort();
    }

    std::string objectText = json.substr(objectStart, objectEnd - objectStart);

    TileOffloadKernelDesc desc;

    if (!jsonFindInt(objectText, "id", desc.id)) {
      pos = objectEnd;
      continue;
    }

    if (!jsonFindString(objectText, "name", desc.name))
      desc.name = "tileoff_kernel_" + std::to_string(desc.id);

    jsonFindString(objectText, "kind", desc.kind);
    jsonFindString(objectText, "backend", desc.backend);
    jsonFindString(objectText, "accelerator_target", desc.acceleratorTarget);
    jsonFindString(objectText, "device_image_kind", desc.deviceImageKind);

    if (!jsonFindInt(objectText, "image_index", desc.ptxIndex))
      jsonFindInt(objectText, "ptx_index", desc.ptxIndex);
    if (!jsonFindString(objectText, "image_file", desc.ptxFile))
      jsonFindString(objectText, "ptx_file", desc.ptxFile);

    if (desc.ptxFile.empty())
      desc.ptxFile = desc.name +
          (desc.deviceImageKind == "cubin"          ? ".cubin"
                  : desc.deviceImageKind == "hsaco" ? ".hsaco"
                                                    : ".ptx");

    if (desc.id < 0 || desc.ptxIndex < 0) {
      std::fprintf(stderr,
          "TileOffload error: kernel JSON contains a negative id or image index\n");
      std::abort();
    }

    if (desc.deviceImageKind != "ptx" && desc.deviceImageKind != "cubin" &&
        desc.deviceImageKind != "hsaco") {
      std::fprintf(stderr,
          "TileOffload error: kernel id %d has unsupported device image kind '%s'\n",
          desc.id, desc.deviceImageKind.c_str());
      std::abort();
    }

    jsonFindInt(objectText, "rank", desc.rank);

    jsonFindIntArray3(objectText, "tile", desc.tileX, desc.tileY, desc.tileZ);
    if (jsonFindKey(objectText, "loop_steps") != std::string::npos &&
        (!jsonFindIntArray3(objectText, "loop_steps", desc.loopStep[0],
            desc.loopStep[1], desc.loopStep[2]))) {
      std::fprintf(stderr, "TileOffload error: invalid loop_steps metadata\n");
      std::abort();
    }

    if (jsonFindKey(objectText, "loop_step_scalar_indices") !=
            std::string::npos &&
        !jsonFindIntArray3(objectText, "loop_step_scalar_indices",
            desc.loopStepScalarIndex[0], desc.loopStepScalarIndex[1],
            desc.loopStepScalarIndex[2])) {
      std::fprintf(stderr, "TileOffload error: invalid loop step scalar metadata\n");
      std::abort();
    }

    jsonFindInt(objectText, "num_warps", desc.numWarps);
    jsonFindInt(objectText, "threads_per_warp", desc.threadsPerWarp);
    jsonFindInt(objectText, "num_ctas", desc.numCTAs);
    jsonFindInt(objectText, "num_stages", desc.numStages);

    bool hasThreadsPerCTA =
        jsonFindInt(objectText, "threads_per_cta", desc.cudaThreadsPerCTA);
    if (!hasThreadsPerCTA)
      hasThreadsPerCTA = jsonFindInt(
          objectText, "cuda_threads_per_cta", desc.cudaThreadsPerCTA);

    if (!jsonFindInt(
            objectText, "private_pointer_args", desc.tritonHiddenPtrArgs))
      jsonFindInt(
          objectText, "triton_hidden_ptr_args", desc.tritonHiddenPtrArgs);
    jsonFindInt(objectText, "launch_abi_version", desc.launchAbiVersion);
    jsonFindInt(objectText, "array_count", desc.arrayCount);
    jsonFindInt(objectText, "scalar_count", desc.scalarCount);
    jsonFindInt(objectText, "output_count", desc.outputCount);
    jsonFindBool(objectText, "copy_back_writes", desc.copyBackWrites);

    desc.pack = jsonParsePackEntries(objectText);
    desc.parameters = jsonParseParameterEntries(objectText);
    jsonFindInt(objectText, "reduction_stage_id", desc.reductionStageId);

    std::string reductionOp;
    if (jsonFindString(objectText, "reduction_op", reductionOp)) {
      if (reductionOp == "add")
        desc.reductionOp = TileOffloadKernelDesc::ReductionOperator::Add;
      else if (reductionOp == "multiply")
        desc.reductionOp = TileOffloadKernelDesc::ReductionOperator::Multiply;
      else if (reductionOp == "min")
        desc.reductionOp = TileOffloadKernelDesc::ReductionOperator::Min;
      else if (reductionOp == "max")
        desc.reductionOp = TileOffloadKernelDesc::ReductionOperator::Max;
      else {
        std::fprintf(stderr,
            "TileOffload error: kernel id %d has unknown reduction_op '%s'\n",
            desc.id, reductionOp.c_str());
        std::abort();
      }
    }

    int64_t expectedCudaThreads =
        static_cast<int64_t>(desc.numWarps) * desc.threadsPerWarp;
    bool supportedSubgroupWidth = desc.threadsPerWarp == 32;
#if defined(TILEOFFLOAD_RUNTIME_USE_HIP) && TILEOFFLOAD_RUNTIME_USE_HIP
    supportedSubgroupWidth =
        supportedSubgroupWidth || desc.threadsPerWarp == 64;
#endif
    if (desc.rank < 1 || desc.rank > 3 || desc.tileX <= 0 || desc.tileY <= 0 ||
        desc.tileZ <= 0 || desc.numWarps <= 0 || !supportedSubgroupWidth ||
        desc.numCTAs <= 0 || desc.numStages <= 0 || expectedCudaThreads <= 0 ||
        expectedCudaThreads > std::numeric_limits<int32_t>::max()) {
      std::fprintf(stderr,
          "TileOffload error: invalid launch metadata for kernel id %d\n", desc.id);
      std::abort();
    }

    if (desc.backend == "cuda-tile")
      expectedCudaThreads = 1;
    if (!hasThreadsPerCTA)
      desc.cudaThreadsPerCTA = static_cast<int32_t>(expectedCudaThreads);
    if (desc.cudaThreadsPerCTA != expectedCudaThreads) {
      std::fprintf(stderr,
          "TileOffload error: threads_per_cta disagrees with subgroup metadata "
          "for kernel id %d\n",
          desc.id);
      std::abort();
    }
    if (desc.tritonHiddenPtrArgs != (desc.backend == "cuda-tile" ? 0 : 2)) {
      std::fprintf(stderr,
          "TileOffload error: kernel id %d requires %d private pointer parameters; "
          "private argument count disagrees with backend\n",
          desc.id, desc.tritonHiddenPtrArgs);
      std::abort();
    }
#if defined(TILEOFFLOAD_RUNTIME_USE_HIP) && TILEOFFLOAD_RUNTIME_USE_HIP
    if (desc.acceleratorTarget != "hip" || desc.deviceImageKind != "hsaco") {
      std::fprintf(stderr,
          "TileOffload error: HIP runtime cannot load target='%s' image_kind='%s' "
          "for kernel id %d\n",
          desc.acceleratorTarget.c_str(), desc.deviceImageKind.c_str(),
          desc.id);
      std::abort();
    }
#else
    if (desc.acceleratorTarget != "cuda" ||
        (desc.deviceImageKind != "ptx" && desc.deviceImageKind != "cubin")) {
      std::fprintf(stderr,
          "TileOffload error: CUDA runtime cannot load target='%s' image_kind='%s' "
          "for kernel id %d\n",
          desc.acceleratorTarget.c_str(), desc.deviceImageKind.c_str(),
          desc.id);
      std::abort();
    }
#endif
    if (desc.kind == "stencil2d" && desc.launchAbiVersion != 2) {
      std::fprintf(stderr,
          "TileOffload error: invalid stencil2d ABI metadata for kernel id %d\n",
          desc.id);
      std::abort();
    }
    if (desc.launchAbiVersion == 2) {
      TileOffloadValidateVariadicKernelMetadata(desc);
      TileOffloadPrepareVariadicMetadata(desc);
    }

    for (const auto &entry : result) {
      if (entry.second.name == desc.name) {
        std::fprintf(stderr, "TileOffload error: duplicate kernel name '%s'\n",
            desc.name.c_str());
        std::abort();
      }
    }
    if (!result.emplace(desc.id, std::move(desc)).second) {
      std::fprintf(stderr, "TileOffload error: duplicate kernel id in JSON\n");
      std::abort();
    }

    pos = objectEnd;
  }

  return result;
}

static void TileOffloadCleanup() {
  TILEOFF_REGISTRY_GUARD();
  if (!TileOffloadRegistry.initialized)
    return;

  if (TileOffloadReductionStatsEnabled())
    TileOffloadPrintReductionWorkspaceStats();

  for (auto &entry : TileOffloadRegistry.contexts) {
    TileOffloadContextState &state = entry.second;
    TileOffloadReductionWorkspace &workspace = state.reductionWorkspace;
    if (state.context)
      cuCtxSetCurrent(state.context);
    if (state.stream)
      cuStreamSynchronize(state.stream);
    if (workspace.partials.ptr)
      cuMemFree(workspace.partials.ptr);
    if (workspace.scratch.ptr)
      cuMemFree(workspace.scratch.ptr);
    if (workspace.results.ptr)
      cuMemFree(workspace.results.ptr);

    for (auto &allocation : state.deviceCache)
      if (allocation.second.ptr)
        cuMemFree(allocation.second.ptr);
    state.deviceCache.clear();
    state.dataRegions.clear();
    state.functionCache.clear();
    state.functionProperties.clear();
    for (CUmodule module : state.modules)
      if (module)
        cuModuleUnload(module);
    state.modules.clear();
    if (state.completionEvent)
      cuEventDestroy(state.completionEvent);
    if (state.stream)
      cuStreamDestroy(state.stream);
  }

  for (auto &entry : TileOffloadRegistry.contexts)
    if (entry.second.retainedPrimaryContext)
      cuDevicePrimaryCtxRelease(entry.second.device);

  TileOffloadRegistry.contexts.clear();
  TileOffloadRegistry.primaryContexts.clear();
  TileOffloadActiveContext = nullptr;
  TileOffloadRegistry.ptxTexts.clear();
  TileOffloadRegistry.embeddedImageKinds.clear();
  TileOffloadRegistry.kernels.clear();

  TileOffloadRegistry.initialized = false;
}

static unsigned TileOffloadReductionIntegerDynamicSharedBytes(
    const TileOffloadKernelDesc *desc, int32_t blockX, size_t integerSize) {
  if (desc && desc->backend == "cuda-tile")
    return 0;

  std::size_t bytes = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      integerSize, "reduction dynamic shared bytes");

  // Align to 256 bytes.
  bytes = TileOffloadCheckedAdd(bytes, 255, "reduction shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: reduction dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
}

static unsigned TileOffloadReductionDynamicSharedBytes(
    const TileOffloadKernelDesc *desc, int32_t blockX) {
  if (desc && desc->backend == "cuda-tile")
    return 0;

  std::size_t bytes = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      sizeof(float), "reduction dynamic shared bytes");

  // Align to 256 bytes.
  bytes = TileOffloadCheckedAdd(bytes, 255, "reduction shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: reduction dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
}

static unsigned TileOffloadReductionF64DynamicSharedBytes(
    const TileOffloadKernelDesc *desc, int32_t blockX) {
  if (desc && desc->backend == "cuda-tile")
    return 0;

  std::size_t bytes = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      sizeof(double), "reduction f64 dynamic shared bytes");

  // Align to 256 bytes.
  bytes = TileOffloadCheckedAdd(bytes, 255, "reduction f64 shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: reduction f64 dynamic shared memory requirement too "
        "large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
}

static unsigned TileOffloadMatmulDynamicSharedBytes(const TileOffloadKernelDesc *desc,
    int32_t blockX, int32_t blockY, int32_t blockK) {
  int32_t stages = 1;
  if (desc && desc->numStages > 0)
    stages = desc->numStages;

  std::size_t aElems = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockK),
      "matmul dynamic shared A tile elements");

  std::size_t bElems = TileOffloadCheckedMul(static_cast<std::size_t>(blockK),
      static_cast<std::size_t>(blockY),
      "matmul dynamic shared B tile elements");

  std::size_t elems = TileOffloadCheckedMul(
      TileOffloadCheckedAdd(aElems, bElems, "matmul shared tile elements"),
      static_cast<std::size_t>(stages),
      "matmul dynamic shared staged tile elements");

  std::size_t bytes =
      TileOffloadCheckedMul(elems, sizeof(float), "matmul dynamic shared bytes");

  // Align to 256 bytes.
  bytes = TileOffloadCheckedAdd(bytes, 255, "matmul shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple A/B tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: matmul dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  // Optional override for experiments, but do not allow values below the
  // computed requirement. A too-small dynamic shared memory size can cause
  // illegal GPU memory accesses.
  if (const char *value = std::getenv("TILEOFF_MATMUL_SHARED_BYTES")) {
    if (value[0] != '\0') {
      unsigned requested = TileOffloadGetEnvUnsignedAllowZero(
          "TILEOFF_MATMUL_SHARED_BYTES", requiredBytes);

      if (requested < requiredBytes) {
        std::fprintf(stderr,
            "TileOffload error: TILEOFF_MATMUL_SHARED_BYTES=%u is smaller than the "
            "computed required minimum %u bytes for tile=(%d,%d,%d), "
            "num_stages=%d. Refusing to launch because this can cause "
            "CUDA_ERROR_ILLEGAL_ADDRESS.\n",
            requested, requiredBytes, blockX, blockY, blockK, stages);
        std::abort();
      }

      return requested;
    }
  }

  return requiredBytes;
}

static unsigned TileOffloadMatmulF64DynamicSharedBytes(
    const TileOffloadKernelDesc * /*desc*/, int32_t blockX, int32_t blockY,
    int32_t blockK) {
  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid f64 matmul tile shape (%d,%d,%d) while "
        "computing dynamic shared memory\n",
        blockX, blockY, blockK);
    std::abort();
  }

  // The f64 blocked fallback materialises/reduces an M x N x K product-like
  // tensor and Triton lowering uses dynamic shared memory for parts of the
  // lowering. This is a conservative estimate; the optional environment
  // override below can be used to tune/debug.
  std::size_t aElems = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockK), "f64 matmul shared A tile elements");

  std::size_t bElems = TileOffloadCheckedMul(static_cast<std::size_t>(blockK),
      static_cast<std::size_t>(blockY), "f64 matmul shared B tile elements");

  std::size_t prodElems =
      TileOffloadCheckedMul(TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
                          static_cast<std::size_t>(blockY),
                          "f64 matmul shared product M*N elements"),
          static_cast<std::size_t>(blockK),
          "f64 matmul shared product M*N*K elements");

  std::size_t accElems = TileOffloadCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockY), "f64 matmul shared accumulator elems");

  std::size_t elems = TileOffloadCheckedAdd(
      TileOffloadCheckedAdd(aElems, bElems, "f64 matmul A/B shared elements"),
      TileOffloadCheckedAdd(prodElems, accElems,
          "f64 matmul product/accumulator shared elements"),
      "f64 matmul total shared elements");

  std::size_t bytes =
      TileOffloadCheckedMul(elems, sizeof(double), "f64 matmul dynamic shared bytes");

  // Align to 256 bytes.
  bytes = TileOffloadCheckedAdd(bytes, 255, "f64 matmul shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton-generated kernels often assume a non-trivial shared-memory arena.
  // Keep a conservative minimum.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: f64 matmul dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  // Debug/tuning override. Do not allow values below the computed minimum.
  if (const char *value = std::getenv("TILEOFF_MATMUL_F64_SHARED_BYTES")) {
    if (value[0] != '\0') {
      unsigned requested = TileOffloadGetEnvUnsignedAllowZero(
          "TILEOFF_MATMUL_F64_SHARED_BYTES", requiredBytes);

      if (requested < requiredBytes) {
        std::fprintf(stderr,
            "TileOffload error: TILEOFF_MATMUL_F64_SHARED_BYTES=%u is smaller than "
            "the computed required minimum %u bytes for tile=(%d,%d,%d). "
            "Refusing to launch because this can cause illegal GPU memory "
            "accesses.\n",
            requested, requiredBytes, blockX, blockY, blockK);
        std::abort();
      }

      return requested;
    }
  }

  return requiredBytes;
}

static void TileOffloadEnsureInitialized() {
  if (TileOffloadRegistry.initialized)
    return;

  if (TileOffloadDebugEnabled())
    std::fprintf(
        stderr, "TileOffload: runtime build id: %s\n", TILEOFF_RUNTIME_BUILD_ID);

  TILEOFF_CUDA_CHECK(cuInit(0));
  auto parseOneJson = [&](const std::string &json, int32_t imageBase) {
    int32_t schemaVersion = 0;
    if (!jsonFindInt(json, "tileoff_schema_version", schemaVersion)) {
      std::fprintf(
          stderr, "TileOffload error: kernel JSON is missing tileoff_schema_version\n");
      std::abort();
    }
    if (schemaVersion != TILEOFF_SUPPORTED_SCHEMA_VERSION) {
      std::fprintf(stderr,
          "TileOffload error: unsupported kernel JSON schema version %d; "
          "runtime supports version %d\n",
          schemaVersion, TILEOFF_SUPPORTED_SCHEMA_VERSION);
      std::abort();
    }

    auto kernels = TileOffloadParseKernelDescsFromJson(json);
    for (auto &entry : kernels) {
      TileOffloadKernelDesc &desc = entry.second;
      desc.ptxIndex += imageBase;
      for (const auto &existing : TileOffloadRegistry.kernels) {
        if (existing.first == desc.id || existing.second.name == desc.name) {
          std::fprintf(stderr,
              "TileOffload error: embedded bundles contain colliding kernel "
              "identity id=%d name='%s'\n",
              desc.id, desc.name.c_str());
          std::abort();
        }
      }
      TileOffloadRegistry.kernels.emplace(entry.first, std::move(desc));
    }
  };

  const char *jsonOverride = std::getenv("TILEOFF_KERNELS_JSON");
  bool hasJsonOverride = jsonOverride && jsonOverride[0] != '\0';
  bool useEmbeddedBundles = !hasJsonOverride && TileOffloadHasEmbeddedBundles();
  bool useLegacySidecars = !hasJsonOverride && !useEmbeddedBundles &&
      TileOffloadTextFileExists("tileoff_kernels.json");
  if (useEmbeddedBundles) {
    int32_t imageBase = 0;
    for (const TileOffloadEmbeddedKernelBundle &bundle :
        TileOffloadGetEmbeddedKernelBundles()) {
      parseOneJson(std::string(bundle.jsonData, bundle.jsonSize), imageBase);
      imageBase += static_cast<int32_t>(bundle.imageData.size());
    }
  } else if (hasJsonOverride || useLegacySidecars) {
    parseOneJson(TileOffloadGetJsonText(), 0);
  } else if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: no kernel metadata found; initializing data-only runtime\n");
  }

  if (!TileOffloadRegistry.kernels.empty()) {
    TileOffloadRegistry.ptxTexts = TileOffloadGetPtxTexts(TileOffloadRegistry.kernels);
    if (useEmbeddedBundles)
      TileOffloadRegistry.embeddedImageKinds =
          TileOffloadGetImageKindsFromEmbeddedBundles();
  }

  if (!TileOffloadRegistry.kernels.empty() && TileOffloadRegistry.ptxTexts.empty()) {
    std::fprintf(stderr, "TileOffload error: no device images were available\n");
    std::abort();
  }

  for (std::size_t i = 0; i < TileOffloadRegistry.ptxTexts.size(); ++i) {
    if (TileOffloadRegistry.ptxTexts[i].empty()) {
      std::fprintf(stderr,
          "TileOffload error: device image entry %zu is empty or missing\n", i);
      std::abort();
    }
  }

  if (!TileOffloadRegistry.embeddedImageKinds.empty()) {
    if (TileOffloadRegistry.embeddedImageKinds.size() !=
        TileOffloadRegistry.ptxTexts.size()) {
      std::fprintf(stderr,
          "TileOffload error: embedded image-kind table has the wrong size\n");
      std::abort();
    }
    for (const auto &entry : TileOffloadRegistry.kernels) {
      const TileOffloadKernelDesc &desc = entry.second;
      int32_t expectedKind = desc.deviceImageKind == "ptx"
          ? TileOffloadEmbeddedKernelBundle::PTX
          : desc.deviceImageKind == "cubin" ? TileOffloadEmbeddedKernelBundle::Cubin
                                            : TileOffloadEmbeddedKernelBundle::HSACO;
      if (static_cast<std::size_t>(desc.ptxIndex) >=
              TileOffloadRegistry.embeddedImageKinds.size() ||
          TileOffloadRegistry.embeddedImageKinds[static_cast<std::size_t>(
              desc.ptxIndex)] != expectedKind) {
        std::fprintf(stderr,
            "TileOffload error: embedded image kind disagrees with metadata for "
            "kernel id %d\n",
            desc.id);
        std::abort();
      }
    }
  }

  if (TileOffloadDebugEnabled()) {
    for (const auto &entry : TileOffloadRegistry.kernels) {
      const TileOffloadKernelDesc &desc = entry.second;

      std::fprintf(stderr,
          "TileOffload: registered kernel id %d -> '%s' "
          "kind=%s rank=%d tile=(%d,%d,%d) "
          "warps=%d threads_per_warp=%d "
          "cuda_threads_per_cta=%d hidden_ptr_args=%d "
          "backend=%s image_kind=%s reduction_stage_id=%d "
          "image_index=%d image_file=%s\n",
          desc.id, desc.name.c_str(), desc.kind.c_str(), desc.rank, desc.tileX,
          desc.tileY, desc.tileZ, desc.numWarps, desc.threadsPerWarp,
          desc.cudaThreadsPerCTA, desc.tritonHiddenPtrArgs,
          desc.backend.c_str(), desc.deviceImageKind.c_str(),
          desc.reductionStageId, desc.ptxIndex, desc.ptxFile.c_str());
    }
  }

  TileOffloadRegistry.initialized = true;
  std::atexit(TileOffloadCleanup);
}

static TileOffloadContextState &TileOffloadCreateContextState(
    CUdevice device, CUcontext context, bool retainedPrimaryContext) {
  TileOffloadContextState state;
  state.device = device;
  state.context = context;
  state.retainedPrimaryContext = retainedPrimaryContext;
  TILEOFF_CUDA_CHECK(cuCtxSetCurrent(state.context));
  TILEOFF_CUDA_CHECK(cuStreamCreate(&state.stream, CU_STREAM_DEFAULT));
  TILEOFF_CUDA_CHECK(
      cuEventCreate(&state.completionEvent, CU_EVENT_DISABLE_TIMING));

  state.modules.resize(TileOffloadRegistry.ptxTexts.size(), nullptr);
  for (std::size_t i = 0; i < TileOffloadRegistry.ptxTexts.size(); ++i) {
    TILEOFF_CUDA_CHECK(cuModuleLoadDataEx(&state.modules[i],
        TileOffloadRegistry.ptxTexts[i].c_str(), 0, nullptr, nullptr));
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr,
          "TileOffload: loaded CUDA module %zu on device=%d context=%p\n", i,
          static_cast<int>(device), static_cast<void *>(state.context));
  }

  auto inserted = TileOffloadRegistry.contexts.emplace(context, std::move(state));
  if (!inserted.second) {
    std::fprintf(stderr,
        "TileOffload error: duplicate CUDA context state for context %p\n",
        static_cast<void *>(context));
    std::abort();
  }
  return inserted.first->second;
}

static TileOffloadContextState &TileOffloadGetOrCreatePrimaryContextState(int ordinal) {
  auto known = TileOffloadRegistry.primaryContexts.find(ordinal);
  if (known != TileOffloadRegistry.primaryContexts.end())
    return TileOffloadRegistry.contexts.at(known->second);

  CUdevice device = 0;
  CUcontext context = nullptr;
  TILEOFF_CUDA_CHECK(cuDeviceGet(&device, ordinal));
  TILEOFF_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));

  // A primary context may already have been registered through caller-owned
  // mode. Retaining it must not create a duplicate state or a second stream.
  auto existing = TileOffloadRegistry.contexts.find(context);
  if (existing != TileOffloadRegistry.contexts.end()) {
    existing->second.retainedPrimaryContext = true;
    TileOffloadRegistry.primaryContexts.emplace(ordinal, context);
    return existing->second;
  }
  TileOffloadContextState &state =
      TileOffloadCreateContextState(device, context, /*retainedPrimaryContext=*/true);
  TileOffloadRegistry.primaryContexts.emplace(ordinal, context);
  return state;
}

static void TileOffloadEnsureCurrentContext() {
  // The outer operation protects registry selection, then transfers ownership
  // to the selected context lock. Nested delegates reuse that selection.
  if (TileOffloadContextScope && TileOffloadContextScope->isSelected())
    return;
  TileOffloadEnsureInitialized();

  if (TileOffloadSelectedDeviceOrdinal < 0 &&
      TileOffloadEnvFlagEnabled("TILEOFF_USE_CURRENT_CONTEXT")) {
    CUcontext context = nullptr;
    CUdevice device = 0;
    TILEOFF_CUDA_CHECK(cuCtxGetCurrent(&context));
    if (!context) {
      std::fprintf(stderr,
          "TileOffload error: TILEOFF_USE_CURRENT_CONTEXT is set but no CUDA context "
          "is current\n");
      std::abort();
    }
    TILEOFF_CUDA_CHECK(cuCtxGetDevice(&device));
    auto known = TileOffloadRegistry.contexts.find(context);
    TileOffloadContextState &state = known != TileOffloadRegistry.contexts.end()
        ? known->second
        : TileOffloadCreateContextState(
              device, context, /*retainedPrimaryContext=*/false);
    TileOffloadOperation->select(state);
    if (TileOffloadContextScope)
      TileOffloadContextScope->select(state.context);
    return;
  }

  int ordinal = TileOffloadGetDeviceOrdinal();
  TileOffloadContextState &state = TileOffloadGetOrCreatePrimaryContextState(ordinal);
  if (!TileOffloadContextScope || TileOffloadContextScope->callerContext() != state.context)
    TILEOFF_CUDA_CHECK(cuCtxSetCurrent(state.context));
  TileOffloadOperation->select(state);
  if (TileOffloadContextScope)
    TileOffloadContextScope->select(state.context);

  if (TileOffloadDebugEnabled())
    std::fprintf(stderr,
        "TileOffload: active CUDA device ordinal=%d device=%d context=%p\n", ordinal,
        static_cast<int>(state.device), static_cast<void *>(state.context));
}

// Queries initialize only the device driver: no kernel metadata, streams or
// primary contexts are loaded until the first TileOffload data operation or launch.
static int TileOffloadVisibleDeviceCount() {
  CUresult status = cuInit(0);
  if (status == CUDA_ERROR_NO_DEVICE)
    return 0;
  TILEOFF_CUDA_CHECK(status);
  int count = 0;
  status = cuDeviceGetCount(&count);
  if (status == CUDA_ERROR_NO_DEVICE)
    return 0;
  TILEOFF_CUDA_CHECK(status);
  return count;
}

static void TileOffloadValidateDeviceOrdinal(int ordinal, int count) {
  if (ordinal < 0 || ordinal >= count) {
    std::fprintf(stderr,
        "TileOffload error: device number %d is outside the visible range [0,%d)\n",
        ordinal, count);
    std::abort();
  }
}

extern "C" int32_t tileoff_get_num_devices() {
  TILEOFF_RUNTIME_GUARD();
  return TileOffloadVisibleDeviceCount();
}

extern "C" void tileoff_set_device_num(int32_t device) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadValidateDeviceOrdinal(device, TileOffloadVisibleDeviceCount());
  TileOffloadSelectedDeviceOrdinal = device;
}

extern "C" int32_t tileoff_get_device_num() {
  TILEOFF_RUNTIME_GUARD();
  int count = TileOffloadVisibleDeviceCount();
  if (TileOffloadSelectedDeviceOrdinal < 0 &&
      TileOffloadEnvFlagEnabled("TILEOFF_USE_CURRENT_CONTEXT")) {
    CUcontext context = nullptr;
    TILEOFF_CUDA_CHECK(cuCtxGetCurrent(&context));
    if (!context) {
      std::fprintf(stderr,
          "TileOffload error: TILEOFF_USE_CURRENT_CONTEXT is set but no CUDA context "
          "is current\n");
      std::abort();
    }
    CUdevice device;
    TILEOFF_CUDA_CHECK(cuCtxGetDevice(&device));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      CUdevice candidate;
      TILEOFF_CUDA_CHECK(cuDeviceGet(&candidate, ordinal));
      if (candidate == device)
        return ordinal;
    }
    std::fprintf(
        stderr, "TileOffload error: current context device is not visible\n");
    std::abort();
  }
  int ordinal = TileOffloadGetDeviceOrdinal();
  TileOffloadValidateDeviceOrdinal(ordinal, count);
  return ordinal;
}

static void TileOffloadWaitForStream(CUstream stream, CUevent completionEvent) {
  if (!stream || !completionEvent) {
    std::fprintf(stderr, "TileOffload error: runtime stream is not initialized\n");
    std::abort();
  }
  TILEOFF_CUDA_CHECK(cuEventRecord(completionEvent, stream));
  TILEOFF_CUDA_CHECK(cuEventSynchronize(completionEvent));
}

static void TileOffloadWaitForRuntimeStream() {
  TileOffloadContextState &state = TileOffloadActiveContextState();
  TileOffloadWaitForStream(state.stream, state.completionEvent);
  state.pendingResidentLaunches = false;
}

static void TileOffloadSynchronizeActiveContext() {
  if (!TileOffloadActiveContext)
    return;

  TileOffloadContextState &state{TileOffloadActiveContextState()};
  if (state.stream) {
    TileOffloadWaitForStream(state.stream, state.completionEvent);
    state.pendingResidentLaunches = false;
  }
}

// All host transfers must order against the non-default runtime stream. This
// also covers UPDATE DEVICE overwriting an input of a queued resident launch.
static void TileOffloadWaitForPendingResidentLaunches() {
  if (TileOffloadActiveContextState().pendingResidentLaunches)
    TileOffloadWaitForRuntimeStream();
}

static CUresult TileOffloadMemcpyHtoD(
    CUdeviceptr dst, const void *src, std::size_t bytes) {
  TileOffloadWaitForPendingResidentLaunches();
  return cuMemcpyHtoD(dst, src, bytes);
}

static CUresult TileOffloadMemcpyDtoH(void *dst, CUdeviceptr src, std::size_t bytes) {
  TileOffloadWaitForPendingResidentLaunches();
  return cuMemcpyDtoH(dst, src, bytes);
}

// Opt-in throughput mode. Default launches still complete synchronously.
// Only allocations retained in the runtime cache may outlive a launch call;
// host outputs, temporaries, and scalar reductions retain their waits.
static void TileOffloadCompleteArrayLaunch(bool allArgumentsCached) {
  if (allArgumentsCached && TileOffloadEnvFlagEnabled("TILEOFF_ASYNC_RESIDENT")) {
    TileOffloadActiveContextState().pendingResidentLaunches = true;
    return;
  }
  TileOffloadWaitForRuntimeStream();
}

// Host-only helper: validate before dividing, including empty domains.
// No CUDA initialization or launch-state mutation is needed.
extern "C" void __tileoff_trip_count_i32(
    int32_t lower, int32_t upper, int32_t step, int32_t *result) {
  if (!result || step == 0) {
    std::fprintf(stderr, "TileOffload error: runtime loop step must be nonzero\n");
    std::abort();
  }
  int64_t distance = step > 0 ? int64_t(upper) - lower : int64_t(lower) - upper;
  int64_t magnitude = step > 0 ? int64_t(step) : -int64_t(step);
  int64_t count = distance < 0 ? 0 : distance / magnitude + 1;
  if (count > std::numeric_limits<int32_t>::max()) {
    std::fprintf(stderr,
        "TileOffload error: runtime loop trip count exceeds signed 32-bit range\n");
    std::abort();
  }
  *result = static_cast<int32_t>(count);
}

extern "C" void __tileoff_wait() {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (TileOffloadDebugEnabled())
    std::fprintf(stderr, "TileOffload: wait for active runtime stream\n");

  TileOffloadWaitForRuntimeStream();
}

static TileOffloadReductionWorkspace &TileOffloadGetReductionWorkspace() {
  TileOffloadEnsureCurrentContext();
  TileOffloadContextState &state = TileOffloadActiveContextState();
  TileOffloadReductionWorkspace &workspace = state.reductionWorkspace;
  if (!workspace.context) {
    workspace.device = state.device;
    workspace.context = state.context;
  }
  return workspace;
}

static const TileOffloadKernelDesc *TileOffloadLookupKernelDesc(int32_t kernelId) {
  TileOffloadEnsureCurrentContext();

  auto it = TileOffloadRegistry.kernels.find(kernelId);
  if (it == TileOffloadRegistry.kernels.end())
    return nullptr;

  return &it->second;
}

static int32_t TileOffloadTritonHiddenPtrArgCount(int32_t kernelId) {
  if (const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId))
    return desc->tritonHiddenPtrArgs;

  return 2;
}

static void TileOffloadValidateSupportedHiddenPtrArgCount(
    int32_t kernelId, const TileOffloadKernelDesc *desc = nullptr) {
  int32_t count =
      desc ? desc->tritonHiddenPtrArgs : TileOffloadTritonHiddenPtrArgCount(kernelId);

  if (!desc)
    desc = TileOffloadLookupKernelDesc(kernelId);
  if (count == (desc && desc->backend == "cuda-tile" ? 0 : 2))
    return;

  std::fprintf(stderr,
      "TileOffload error: kernel id %d requires %d Triton hidden pointer "
      "but this runtime currently supports exactly 2. "
      "This usually means the Triton/PTX generation pipeline changed and the "
      "TileOffload runtime ABI must be updated.\n",
      kernelId, count);
  std::abort();
}

static std::optional<int32_t> TileOffloadExplicitPackTargetForSlot(
    const TileOffloadKernelDesc *desc, int32_t slot) {
  if (!desc)
    return std::nullopt;

  for (const TileOffloadPackEntry &entry : desc->pack) {
    if (entry.kernelArgSlot == slot)
      return entry.target;
  }

  return std::nullopt;
}

static bool TileOffloadHostPointerIsPresentOnDevice(void *hostPtr) {
  if (!hostPtr)
    return false;

  auto &cache{TileOffloadActiveContextState().deviceCache};
  auto it{cache.find(hostPtr)};

  return it != cache.end() && it->second.ptr != 0 && it->second.bytes != 0;
}

static int32_t TileOffloadEffectivePackTargetForSlot(
    const TileOffloadKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (auto explicitTarget = TileOffloadExplicitPackTargetForSlot(desc, slot))
    return *explicitTarget;

  // Present-if-cached default:
  //
  // If the user has already created a persistent device allocation with
  // enter data/update device/pack(...:device), then later launches can omit
  // pack(...:device). We use the cached device allocation automatically.
  if (TileOffloadHostPointerIsPresentOnDevice(hostPtr))
    return TILEOFF_PACK_TARGET_DEVICE;

  return TILEOFF_PACK_TARGET_HOST;
}

static int32_t TileOffloadEffectiveWriteTargetForSlot(
    const TileOffloadKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (desc && !desc->copyBackWrites)
    return TILEOFF_PACK_TARGET_DEVICE;
  return TileOffloadEffectivePackTargetForSlot(desc, slot, hostPtr);
}

static const char *TileOffloadPackTargetSourceName(
    const TileOffloadKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (TileOffloadExplicitPackTargetForSlot(desc, slot))
    return "explicit";

  if (TileOffloadHostPointerIsPresentOnDevice(hostPtr))
    return "present";

  return "default-host";
}

static const char *TileOffloadPackTargetName(int32_t target) {
  return target == TILEOFF_PACK_TARGET_DEVICE ? "device" : "host";
}

static TileOffloadDeviceArg TileOffloadMakeTemporaryDeviceBuffer(void *hostPtr,
    std::size_t bytes, bool copyHostToDevice, int32_t slot, const char *role) {
  TileOffloadDeviceArg arg;
  arg.cached = false;
  arg.target = TILEOFF_PACK_TARGET_HOST;
  arg.slot = slot;

  TILEOFF_CUDA_CHECK(cuMemAlloc(&arg.ptr, bytes));

  if (copyHostToDevice)
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(arg.ptr, hostPtr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: temporary device buffer for %s slot %d: "
        "host=%p device=0x%llx bytes=%zu copy_in=%s\n",
        role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes,
        copyHostToDevice ? "yes" : "no");
  }

  return arg;
}

using TileOffloadDeviceCache = std::unordered_map<void *, TileOffloadDeviceAllocation>;
static TileOffloadDeviceArg TileOffloadGetCachedDeviceBuffer(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnMiss, int32_t slot,
    const char *role,
    std::optional<TileOffloadDeviceCache::iterator> known = std::nullopt) {
  TileOffloadDeviceArg arg;
  arg.cached = true;
  arg.target = TILEOFF_PACK_TARGET_DEVICE;
  arg.slot = slot;

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = known ? *known : cache.find(hostPtr);

  bool needAllocate = false;

  if (it == cache.end()) {
    needAllocate = true;
  } else if (it->second.bytes != bytes) {
    if (it->second.dataRegionReferences != 0) {
      std::fprintf(stderr,
          "TileOffload error: cannot resize cached allocation for %s slot %d "
          "while it is owned by %zu data region(s)\n",
          role, slot, it->second.dataRegionReferences);
      std::abort();
    }
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: cache size mismatch for %s slot %d host=%p; "
          "old bytes=%zu new bytes=%zu, reallocating\n",
          role, slot, hostPtr, it->second.bytes, bytes);
    }

    TileOffloadSynchronizeActiveContext();

    TILEOFF_CUDA_CHECK(cuMemFree(it->second.ptr));
    cache.erase(it);
    needAllocate = true;
  }

  if (needAllocate) {
    TileOffloadDeviceAllocation allocation;
    allocation.bytes = bytes;
    TILEOFF_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnMiss)
      TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(allocation.ptr, hostPtr, bytes));

    auto inserted = cache.emplace(hostPtr, allocation);
    arg.ptr = inserted.first->second.ptr;

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: cache miss for %s slot %d target=device: "
          "host=%p device=0x%llx bytes=%zu copy_in=%s\n",
          role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes,
          copyHostToDeviceOnMiss ? "yes" : "no");
    }
  } else {
    arg.ptr = it->second.ptr;

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: cache hit for %s slot %d target=device: "
          "host=%p device=0x%llx bytes=%zu\n",
          role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes);
    }
  }

  return arg;
}

static TileOffloadDeviceArg TileOffloadPrepareReadBuffer(
    void *hostPtr, std::size_t bytes, int32_t target, int32_t slot) {
  if (target == TILEOFF_PACK_TARGET_DEVICE) {
    // Device target means cache/reuse device allocation. Copy in only on miss.
    return TileOffloadGetCachedDeviceBuffer(static_cast<void *>(hostPtr), bytes,
        /*copyHostToDeviceOnMiss=*/true, slot, "read");
  }

  return TileOffloadMakeTemporaryDeviceBuffer(static_cast<void *>(hostPtr), bytes,
      /*copyHostToDevice=*/true, slot, "read");
}

static TileOffloadDeviceArg TileOffloadPrepareWriteBuffer(
    void *hostPtr, std::size_t bytes, int32_t target, int32_t slot) {
  if (target == TILEOFF_PACK_TARGET_DEVICE) {
    // Device target means keep the output allocation cached. No copy-in needed.
    return TileOffloadGetCachedDeviceBuffer(static_cast<void *>(hostPtr), bytes,
        /*copyHostToDeviceOnMiss=*/false, slot, "write");
  }

  return TileOffloadMakeTemporaryDeviceBuffer(static_cast<void *>(hostPtr), bytes,
      /*copyHostToDevice=*/false, slot, "write");
}

// Resolve explicit policy, live presence, and allocation in one cache lookup.
static TileOffloadDeviceArg TileOffloadPrepareArrayBuffer(const TileOffloadKernelDesc *desc,
    int32_t slot, void *host, std::size_t bytes, int32_t flags) {
  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(host);
  const auto &explicitTarget = desc->explicitArrayTargets[slot];
  bool present = it != cache.end() && it->second.ptr && it->second.bytes;
  int32_t target = explicitTarget ? *explicitTarget
      : present                   ? TILEOFF_PACK_TARGET_DEVICE
                                  : TILEOFF_PACK_TARGET_HOST;
  if ((flags & 2) && !desc->copyBackWrites)
    target = TILEOFF_PACK_TARGET_DEVICE;
  bool read = (flags & 1) != 0;
  if (target == TILEOFF_PACK_TARGET_DEVICE)
    return TileOffloadGetCachedDeviceBuffer(
        host, bytes, read, slot, read ? "read" : "write", it);
  return TileOffloadMakeTemporaryDeviceBuffer(
      host, bytes, read, slot, read ? "read" : "write");
}

static void TileOffloadCopyBackWriteBuffer(
    void *hostPtr, const TileOffloadDeviceArg &arg, std::size_t bytes) {
  // Callers invoke this only when host visibility is required.
  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(hostPtr, arg.ptr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: copied write slot %d device=0x%llx -> host=%p "
        "bytes=%zu target=%s\n",
        arg.slot, static_cast<unsigned long long>(arg.ptr),
        static_cast<void *>(hostPtr), bytes, TileOffloadPackTargetName(arg.target));
  }
}

static void TileOffloadReleaseDeviceArg(const TileOffloadDeviceArg &arg) {
  if (!arg.ptr)
    return;

  if (arg.cached)
    return;

  TILEOFF_CUDA_CHECK(cuMemFree(arg.ptr));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: freed temporary device buffer for slot %d "
        "device=0x%llx\n",
        arg.slot, static_cast<unsigned long long>(arg.ptr));
  }
}

// -------------------------------------------------------------------------- //
// CUDA module/function management
// -------------------------------------------------------------------------- //
static void TileOffloadDebugFunctionAttributes(CUfunction fn, int32_t kernelId) {
  if (!TileOffloadDebugEnabled())
    return;

  int maxThreadsPerBlock = 0;
  int numRegs = 0;
  int sharedBytes = 0;
  int binaryVersion = 0;
  int ptxVersion = 0;

  cuFuncGetAttribute(
      &maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn);

  cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);

  cuFuncGetAttribute(&sharedBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);

  cuFuncGetAttribute(&binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, fn);

  cuFuncGetAttribute(&ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, fn);

  std::fprintf(stderr,
      "TileOffload: function attrs for kernel id %d: "
      "max_threads_per_block=%d num_regs=%d shared_bytes=%d "
      "binary_version=%d ptx_version=%d\n",
      kernelId, maxThreadsPerBlock, numRegs, sharedBytes, binaryVersion,
      ptxVersion);
}

static CUfunction getKernelFunction(int32_t kernelId) {
  TileOffloadEnsureCurrentContext();
  TileOffloadContextState &state = TileOffloadActiveContextState();

  auto cacheIt = state.functionCache.find(kernelId);
  if (cacheIt != state.functionCache.end())
    return cacheIt->second;

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  std::string kernelName;

  if (desc) {
    kernelName = desc->name;
  } else {
    std::fprintf(
        stderr, "TileOffload error: no JSON descriptor for kernel id %d\n", kernelId);
    std::abort();
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: looking up CUDA kernel id %d as symbol '%s'\n",
        kernelId, kernelName.c_str());
  }

  std::size_t moduleIndex = 0;
  if (desc)
    moduleIndex = static_cast<std::size_t>(desc->ptxIndex);

  if (moduleIndex >= state.modules.size() || !state.modules[moduleIndex]) {
    std::fprintf(stderr,
        "TileOffload error: kernel id %d requests device image index %zu, "
        "but only %zu module(s) are loaded\n",
        kernelId, moduleIndex, state.modules.size());
    std::abort();
  }

  CUfunction fn = nullptr;
  TILEOFF_CUDA_CHECK(
      cuModuleGetFunction(&fn, state.modules[moduleIndex], kernelName.c_str()));

  state.functionCache[kernelId] = fn;
  return fn;
}

static unsigned TileOffloadCudaThreadsPerCTA(const TileOffloadKernelDesc *desc) {
  if (desc) {
    if (desc->cudaThreadsPerCTA > 0)
      return static_cast<unsigned>(desc->cudaThreadsPerCTA);

    if (desc->numWarps > 0 && desc->threadsPerWarp > 0)
      return static_cast<unsigned>(desc->numWarps * desc->threadsPerWarp);
  }

  return 32;
}

static unsigned TileOffloadCudaThreadsPerCTA(int32_t kernelId) {
  return TileOffloadCudaThreadsPerCTA(TileOffloadLookupKernelDesc(kernelId));
}

static void TileOffloadValidateHostLaunchAgainstDesc(const TileOffloadKernelDesc *desc,
    int32_t kernelId, int32_t rank, int32_t blockX, int32_t blockY,
    int32_t blockZ) {
  if (!desc)
    return;

  if (desc->rank != rank) {
    std::fprintf(stderr,
        "TileOffload error: host launch rank %d disagrees with JSON "
        "rank %d for kernel id %d\n",
        rank, desc->rank, kernelId);
    std::abort();
  }

  if (desc->tileX != blockX || desc->tileY != blockY || desc->tileZ != blockZ) {
    std::fprintf(stderr,
        "TileOffload error: host tile (%d,%d,%d) disagrees with JSON "
        "tile (%d,%d,%d) for kernel id %d\n",
        blockX, blockY, blockZ, desc->tileX, desc->tileY, desc->tileZ,
        kernelId);
    std::abort();
  }
}

static void TileOffloadValidateHostLaunchAgainstDesc(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ) {
  TileOffloadValidateHostLaunchAgainstDesc(
      TileOffloadLookupKernelDesc(kernelId), kernelId, rank, blockX, blockY, blockZ);
}

static unsigned TileOffloadCdiv(std::int64_t x, std::int64_t y, const char *what) {
  if (x < 0 || y <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid values while computing %s: "
        "x=%lld y=%lld\n",
        what, static_cast<long long>(x), static_cast<long long>(y));
    std::abort();
  }

  if (x == 0)
    return 0;

  std::uint64_t numerator{
      static_cast<std::uint64_t>(x) + static_cast<std::uint64_t>(y) - 1};
  std::uint64_t result{numerator / static_cast<std::uint64_t>(y)};

  if (result > std::numeric_limits<unsigned>::max()) {
    std::fprintf(stderr,
        "TileOffload error: grid dimension overflow while computing %s\n", what);
    std::abort();
  }

  return static_cast<unsigned>(result);
}

static std::size_t TileOffloadElementCount(
    int32_t rank, int32_t extentX, int32_t extentY, int32_t extentZ) {
  if (rank < 1 || rank > 3 || extentX < 0 || extentY < 0 || extentZ < 0) {
    std::fprintf(stderr, "TileOffload error: invalid rank or negative extent\n");
    std::abort();
  }

  std::size_t count = static_cast<std::size_t>(extentX);

  if (rank >= 2)
    count = TileOffloadCheckedMul(
        count, static_cast<std::size_t>(extentY), "launch element count");

  if (rank >= 3)
    count = TileOffloadCheckedMul(
        count, static_cast<std::size_t>(extentZ), "launch element count");

  return count;
}

static void TileOffloadValidateCommonLaunchInputs(const char *entryName, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, int32_t extentX, int32_t extentY, int32_t extentZ) {
  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid tile/block shape (%d,%d,%d) in %s\n", blockX,
        blockY, blockZ, entryName);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(
        stderr, "TileOffload error: unsupported rank %d in %s\n", rank, entryName);
    std::abort();
  }

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "TileOffload error: null host pointer in %s: "
        "a=%p b=%p c=%p\n",
        entryName, static_cast<void *>(a), static_cast<void *>(b),
        static_cast<void *>(c));
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: non-positive extent (%d,%d,%d) in %s; "
          "skipping launch\n",
          extentX, extentY, extentZ, entryName);
    }
  }
}

} // namespace

extern "C" void __tileoff_validate_contiguous_desc(void *hostPtr,
    int64_t elementBytes, int32_t rank, int64_t extent0, int64_t extent1,
    int64_t extent2, int64_t stride0, int64_t stride1, int64_t stride2) {
  // Pure descriptor validation needs no runtime lock.
  if (!hostPtr && extent0 != 0 && extent1 != 0 && extent2 != 0) {
    std::fprintf(
        stderr, "TileOffload error: launch descriptor has a null base pointer\n");
    std::abort();
  }
  TileOffloadValidateContiguousDescriptor("TileOffload kernel launch", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);
}

extern "C" void __tileoff_validate_launch_desc(void *hostPtr,
    int64_t elementBytes, int32_t rank, int64_t lower0, int64_t lower1,
    int64_t lower2, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2, int32_t expectedRank,
    int64_t expectedExtent0, int64_t expectedExtent1, int64_t expectedExtent2) {
  // Pure descriptor validation needs no runtime lock.

  TileOffloadValidateContiguousDescriptor("TileOffload kernel launch", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);

  if (rank != expectedRank) {
    std::fprintf(stderr,
        "TileOffload error: descriptor rank %d does not match "
        "kernel rank %d\n",
        rank, expectedRank);
    std::abort();
  }

  const int64_t lowers[3]{lower0, lower1, lower2};
  const int64_t extents[3]{extent0, extent1, extent2};
  const int64_t expected[3]{expectedExtent0, expectedExtent1, expectedExtent2};

  for (int32_t dim = 0; dim < rank; ++dim) {
    if (lowers[dim] != 1) {
      std::fprintf(stderr,
          "TileOffload error: dimension %d has lower bound %lld; "
          "the current Triton lowering requires lower bound 1\n",
          dim + 1, static_cast<long long>(lowers[dim]));
      std::abort();
    }

    if (extents[dim] < expected[dim]) {
      std::fprintf(stderr,
          "TileOffload error: dimension %d extent %lld is smaller "
          "than required launch extent %lld\n",
          dim + 1, static_cast<long long>(extents[dim]),
          static_cast<long long>(expected[dim]));
      std::abort();
    }
  }

  if (!hostPtr && expectedExtent0 > 0 && expectedExtent1 > 0 &&
      expectedExtent2 > 0) {
    std::fprintf(
        stderr, "TileOffload error: nonempty launch has a null array pointer\n");
    std::abort();
  }
}

// -------------------------------------------------------------------------- //
// Public runtime ABI: binary elementwise kernels
// -------------------------------------------------------------------------- //
//
// Current binary kernel signatures:
//
//   rank 1 TTIR:
//     (%a: ptr<f32>, %b: ptr<f32>, %c: ptr<f32>, %n: i32)
//
//   rank 2 TTIR:
//     (%a: ptr<f32>, %b: ptr<f32>, %c: ptr<f32>, %n: i32, %m: i32)
//
static std::size_t TileOffloadElementCountFromExtents(
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2) {
  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "TileOffload error: descriptor data directive received unsupported rank %d\n",
        rank);
    std::abort();
  }

  if (extent0 < 0 || extent1 < 0 || extent2 < 0) {
    std::fprintf(stderr,
        "TileOffload error: descriptor data directive received negative extent "
        "(%lld,%lld,%lld)\n",
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2));
    std::abort();
  }

  std::size_t count = static_cast<std::size_t>(extent0);

  if (rank >= 2)
    count = TileOffloadCheckedMul(
        count, static_cast<std::size_t>(extent1), "descriptor element count");

  if (rank >= 3)
    count = TileOffloadCheckedMul(
        count, static_cast<std::size_t>(extent2), "descriptor element count");

  return count;
}

static std::size_t TileOffloadBytesFromDescriptor(int64_t elementBytes, int32_t rank,
    int64_t extent0, int64_t extent1, int64_t extent2) {
  if (elementBytes <= 0) {
    std::fprintf(stderr,
        "TileOffload error: descriptor data directive received invalid element size "
        "%lld\n",
        static_cast<long long>(elementBytes));
    std::abort();
  }

  std::size_t elements =
      TileOffloadElementCountFromExtents(rank, extent0, extent1, extent2);

  return TileOffloadCheckedMul(
      elements, static_cast<std::size_t>(elementBytes), "descriptor bytes");
}

static TileOffloadDeviceAllocation &TileOffloadGetOrCreateCachedAllocation(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnCreateOrResize,
    const char *operationName) {

  if (!hostPtr) {
    std::fprintf(stderr, "TileOffload error: cannot cache a null host pointer\n");
    std::abort();
  }

  if (bytes == 0) {
    std::fprintf(stderr,
        "TileOffload error: persistent TileOffload allocations must have "
        "nonzero size\n");
    std::abort();
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);

  if (it != cache.end()) {
    if (it->second.bytes == bytes)
      return it->second;

    if (it->second.dataRegionReferences != 0) {
      std::fprintf(stderr,
          "TileOffload error: %s cannot resize host=%p while it is owned by %zu "
          "data region(s)\n",
          operationName, hostPtr, it->second.dataRegionReferences);
      std::abort();
    }

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: %s resizing cached allocation for host=%p "
          "old_bytes=%zu new_bytes=%zu\n",
          operationName, hostPtr, it->second.bytes, bytes);
    }

    TileOffloadSynchronizeActiveContext();

    TILEOFF_CUDA_CHECK(cuMemFree(it->second.ptr));
    cache.erase(it);
  }

  TileOffloadDeviceAllocation allocation;
  allocation.bytes = bytes;

  if (bytes > 0) {
    TILEOFF_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnCreateOrResize)
      TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(allocation.ptr, hostPtr, bytes));
  }

  auto inserted = cache.emplace(hostPtr, allocation);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s created cached allocation host=%p device=0x%llx "
        "bytes=%zu copy_in=%s\n",
        operationName, hostPtr,
        static_cast<unsigned long long>(inserted.first->second.ptr), bytes,
        copyHostToDeviceOnCreateOrResize ? "yes" : "no");
  }

  return inserted.first->second;
}

static TileOffloadDataRegionFrame &TileOffloadCurrentDataRegion(const char *operationName) {
  auto &regions = TileOffloadActiveContextState().dataRegions;
  if (regions.empty()) {
    std::fprintf(stderr,
        "TileOffload error: %s requires an active ENTER DATA region\n",
        operationName);
    std::abort();
  }
  return regions.back();
}

static bool TileOffloadFrameOwnsAllocation(
    const TileOffloadDataRegionFrame &frame, void *hostPtr) {
  return std::find(frame.allocations.begin(), frame.allocations.end(),
             hostPtr) != frame.allocations.end();
}

static TileOffloadDeviceAllocation &TileOffloadAcquireDataRegionAllocation(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnCreate,
    const char *operationName) {
  TileOffloadDataRegionFrame &frame = TileOffloadCurrentDataRegion(operationName);
  auto &cache = TileOffloadActiveContextState().deviceCache;

  if (TileOffloadFrameOwnsAllocation(frame, hostPtr)) {
    auto existing = cache.find(hostPtr);
    if (existing == cache.end()) {
      std::fprintf(stderr,
          "TileOffload error: %s found corrupt data-region ownership for host=%p\n",
          operationName, hostPtr);
      std::abort();
    }
    if (existing->second.bytes != bytes) {
      std::fprintf(stderr,
          "TileOffload error: %s repeats host=%p with a different size inside one "
          "data region (%zu versus %zu bytes)\n",
          operationName, hostPtr, existing->second.bytes, bytes);
      std::abort();
    }
    return existing->second;
  }

  TileOffloadDeviceAllocation *allocation = nullptr;
  if (bytes == 0) {
    auto inserted = cache.emplace(hostPtr, TileOffloadDeviceAllocation{});
    if (!inserted.second && inserted.first->second.bytes != 0) {
      std::fprintf(stderr,
          "TileOffload error: %s cannot acquire zero-sized host=%p while a "
          "non-empty allocation is present\n",
          operationName, hostPtr);
      std::abort();
    }
    allocation = &inserted.first->second;
  } else {
    allocation = &TileOffloadGetOrCreateCachedAllocation(
        hostPtr, bytes, copyHostToDeviceOnCreate, operationName);
  }
  ++allocation->dataRegionReferences;
  frame.allocations.push_back(hostPtr);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s acquired host=%p in data region depth=%zu references=%zu\n",
        operationName, hostPtr, TileOffloadActiveContextState().dataRegions.size(),
        allocation->dataRegionReferences);
  }
  return *allocation;
}

static TileOffloadDeviceAllocation &TileOffloadAcquireExistingDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  TileOffloadDataRegionFrame &frame = TileOffloadCurrentDataRegion(operationName);
  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "TileOffload error: %s has no sized cached allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }

  if (!TileOffloadFrameOwnsAllocation(frame, hostPtr)) {
    ++it->second.dataRegionReferences;
    frame.allocations.push_back(hostPtr);
  }
  return it->second;
}

static TileOffloadDeviceAllocation &TileOffloadGetOwnedDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  TileOffloadDataRegionFrame &frame = TileOffloadCurrentDataRegion(operationName);
  if (!TileOffloadFrameOwnsAllocation(frame, hostPtr)) {
    std::fprintf(stderr,
        "TileOffload error: %s for host=%p does not belong to the innermost data "
        "region\n",
        operationName, hostPtr);
    std::abort();
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end() || it->second.dataRegionReferences == 0) {
    std::fprintf(stderr,
        "TileOffload error: %s found corrupt data-region allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }
  return it->second;
}

static void TileOffloadCopyoutDataRegionAllocation(
    void *hostPtr, std::size_t bytes, const char *operationName) {
  TileOffloadDeviceAllocation &allocation =
      TileOffloadGetOwnedDataRegionAllocation(hostPtr, operationName);
  if (allocation.bytes < bytes) {
    std::fprintf(stderr,
        "TileOffload error: %s requested %zu bytes for host=%p, but the cached "
        "allocation has only %zu bytes\n",
        operationName, bytes, hostPtr, allocation.bytes);
    std::abort();
  }

  // Present-or-copyout semantics: an inner region relinquishes only its own
  // reference. The host is updated only by the last owning region.
  if (allocation.dataRegionReferences == 1) {
    if (bytes != 0)
      TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(hostPtr, allocation.ptr, bytes));
  } else if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s deferred host copy for host=%p; %zu enclosing data "
        "region reference(s) remain\n",
        operationName, hostPtr, allocation.dataRegionReferences - 1);
  }
}

static void TileOffloadReleaseDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  TileOffloadDataRegionFrame &frame = TileOffloadCurrentDataRegion(operationName);
  auto owned =
      std::find(frame.allocations.begin(), frame.allocations.end(), hostPtr);
  if (owned == frame.allocations.end()) {
    std::fprintf(stderr,
        "TileOffload error: %s for host=%p does not belong to the innermost data "
        "region\n",
        operationName, hostPtr);
    std::abort();
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto allocation = cache.find(hostPtr);
  if (allocation == cache.end() ||
      allocation->second.dataRegionReferences == 0) {
    std::fprintf(stderr,
        "TileOffload error: %s found corrupt data-region allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }

  frame.allocations.erase(owned);
  --allocation->second.dataRegionReferences;
  if (allocation->second.dataRegionReferences != 0)
    return;

  CUdeviceptr devicePtr = allocation->second.ptr;
  std::size_t bytes = allocation->second.bytes;
  TileOffloadSynchronizeActiveContext();
  if (devicePtr)
    TILEOFF_CUDA_CHECK(cuMemFree(devicePtr));
  cache.erase(allocation);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s released final data-region allocation host=%p "
        "device=0x%llx bytes=%zu\n",
        operationName, hostPtr, static_cast<unsigned long long>(devicePtr),
        bytes);
  }
}

extern "C" void __tileoff_enter_data_region() {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadActiveContextState().dataRegions.emplace_back();
  if (TileOffloadDebugEnabled())
    std::fprintf(stderr, "TileOffload: enter data region depth=%zu\n",
        TileOffloadActiveContextState().dataRegions.size());
}

extern "C" void __tileoff_exit_data_region() {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  auto &regions = TileOffloadActiveContextState().dataRegions;
  if (regions.empty()) {
    std::fprintf(stderr,
        "TileOffload error: EXIT DATA has no matching active ENTER DATA region\n");
    std::abort();
  }

  while (!regions.back().allocations.empty()) {
    void *hostPtr = regions.back().allocations.back();
    TileOffloadReleaseDataRegionAllocation(hostPtr, "exit_data_region");
  }
  regions.pop_back();

  if (TileOffloadDebugEnabled())
    std::fprintf(stderr, "TileOffload: exit data region depth=%zu\n", regions.size());
}

extern "C" void __tileoff_data_copyin_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: data_copyin_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  TileOffloadAcquireDataRegionAllocation(hostPtr,
      static_cast<std::size_t>(bytesValue),
      /*copyHostToDeviceOnCreate=*/true, "data_copyin_bytes");
}

extern "C" void __tileoff_data_create_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: data_create_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  TileOffloadAcquireDataRegionAllocation(hostPtr,
      static_cast<std::size_t>(bytesValue),
      /*copyHostToDeviceOnCreate=*/false, "data_create_bytes");
}

extern "C" void __tileoff_data_copyout_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: data_copyout_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  TileOffloadCopyoutDataRegionAllocation(
      hostPtr, static_cast<std::size_t>(bytesValue), "data_copyout_bytes");
}

extern "C" void __tileoff_data_copyin_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  TileOffloadValidateContiguousDescriptor("__tileoff_data_copyin_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  TileOffloadAcquireDataRegionAllocation(hostPtr, bytes,
      /*copyHostToDeviceOnCreate=*/true, "data_copyin_desc");
}

extern "C" void __tileoff_data_create_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  TileOffloadValidateContiguousDescriptor("__tileoff_data_create_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  TileOffloadAcquireDataRegionAllocation(hostPtr, bytes,
      /*copyHostToDeviceOnCreate=*/false, "data_create_desc");
}

extern "C" void __tileoff_data_copyout_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  TileOffloadValidateContiguousDescriptor("__tileoff_data_copyout_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  TileOffloadCopyoutDataRegionAllocation(hostPtr, bytes, "data_copyout_desc");
}

extern "C" void __tileoff_data_copyin(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (hostPtr)
    TileOffloadAcquireExistingDataRegionAllocation(hostPtr, "data_copyin");
}

extern "C" void __tileoff_data_copyout(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;
  TileOffloadDeviceAllocation &allocation =
      TileOffloadGetOwnedDataRegionAllocation(hostPtr, "data_copyout");
  TileOffloadCopyoutDataRegionAllocation(hostPtr, allocation.bytes, "data_copyout");
}

extern "C" void __tileoff_data_delete(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  if (!hostPtr)
    return;

  // Validate that DELETE names an allocation owned by this frame. Actual
  // release is performed by the following data-region-exit marker so clause
  // source order cannot make DELETE run before COPYOUT.
  TileOffloadDeviceAllocation &allocation =
      TileOffloadGetOwnedDataRegionAllocation(hostPtr, "data_delete");
  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: data_delete marked host=%p device=0x%llx for release at "
        "region exit (references=%zu)\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        allocation.dataRegionReferences);
  }
}

static void TileOffloadRequirePresentAllocation(
    const char *operationName, void *hostPtr, std::size_t requiredBytes) {
  if (!hostPtr) {
    std::fprintf(stderr, "TileOffload error: %s received a null host pointer\n",
        operationName);
    std::abort();
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end() || it->second.ptr == 0 || it->second.bytes == 0) {
    std::fprintf(stderr,
        "TileOffload error: %s requires host=%p to be present on the device; "
        "use enter data copyin/create first\n",
        operationName, hostPtr);
    std::abort();
  }

  if (requiredBytes != 0 && it->second.bytes < requiredBytes) {
    std::fprintf(stderr,
        "TileOffload error: %s requires %zu bytes for host=%p, but the present "
        "allocation has only %zu bytes\n",
        operationName, requiredBytes, hostPtr, it->second.bytes);
    std::abort();
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s confirmed host=%p device=0x%llx bytes=%zu\n", operationName,
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        it->second.bytes);
  }
}

extern "C" void __tileoff_present(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  TileOffloadRequirePresentAllocation("present", hostPtr, 0);
}

extern "C" void __tileoff_present_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: present_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  if (bytesValue == 0)
    return;

  TileOffloadRequirePresentAllocation(
      "present_bytes", hostPtr, static_cast<std::size_t>(bytesValue));
}

extern "C" void __tileoff_present_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateContiguousDescriptor("__tileoff_present_desc", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  if (bytes == 0)
    return;

  TileOffloadRequirePresentAllocation("present_desc", hostPtr, bytes);
}

extern "C" void __tileoff_create_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: create_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: create_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: create_bytes ignored zero-size object host=%p\n", hostPtr);
    }
    return;
  }

  TileOffloadDeviceAllocation &allocation = TileOffloadGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "create_bytes");

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: create_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(allocation.ptr), bytes);
  }
}

extern "C" void __tileoff_create_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: create_desc ignored null pointer\n");
    return;
  }

  TileOffloadValidateContiguousDescriptor("__tileoff_create_desc", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: create_desc ignored zero-size object host=%p\n", hostPtr);
    }
    return;
  }

  TileOffloadDeviceAllocation &allocation = TileOffloadGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "create_desc");

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: create_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __tileoff_update_device_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;

  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_device_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: update_device_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: update_device_bytes ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  TileOffloadDeviceAllocation &allocation = TileOffloadGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "update_device_bytes");

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(allocation.ptr, hostPtr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: update_device_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(allocation.ptr), bytes);
  }
}

extern "C" void __tileoff_update_host_bytes(void *hostPtr, int64_t bytesValue) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_host_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "TileOffload error: update_host_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: update_host_bytes ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "TileOffload error: update_host_bytes has no cached allocation for "
        "host=%p bytes=%zu; use create/copyin/update_device first\n",
        hostPtr, bytes);
    std::abort();
  }

  if (it->second.bytes < bytes) {
    std::fprintf(stderr,
        "TileOffload error: update_host_bytes requested %zu bytes for host=%p, "
        "but cached allocation has only %zu bytes\n",
        bytes, hostPtr, it->second.bytes);
    std::abort();
  }

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(hostPtr, it->second.ptr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: update_host_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(it->second.ptr), bytes);
  }
}

extern "C" void __tileoff_update_device_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_device_desc ignored null pointer\n");
    return;
  }

  TileOffloadValidateContiguousDescriptor("__tileoff_update_device_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: update_device_desc ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  TileOffloadDeviceAllocation &allocation = TileOffloadGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "update_device_desc");

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(allocation.ptr, hostPtr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: update_device_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __tileoff_update_host_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_host_desc ignored null pointer\n");
    return;
  }

  TileOffloadValidateContiguousDescriptor("__tileoff_update_host_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      TileOffloadBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: update_host_desc ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "TileOffload error: update_host_desc has no cached allocation for host=%p; "
        "use create/copyin/update_device first\n",
        hostPtr);
    std::abort();
  }

  if (it->second.bytes < bytes) {
    std::fprintf(stderr,
        "TileOffload error: update_host_desc requested %zu bytes for host=%p, "
        "but cached allocation has only %zu bytes\n",
        bytes, hostPtr, it->second.bytes);
    std::abort();
  }

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(hostPtr, it->second.ptr, bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: update_host_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __tileoff_release_desc(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: release_desc ignored null pointer\n");
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: release_desc ignored; no cached allocation for host=%p\n",
          hostPtr);
    }
    return;
  }

  if (it->second.dataRegionReferences != 0) {
    std::fprintf(stderr,
        "TileOffload error: release_desc cannot release host=%p while it is owned "
        "by %zu data region(s); exit the owning region first\n",
        hostPtr, it->second.dataRegionReferences);
    std::abort();
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  TileOffloadSynchronizeActiveContext();

  TILEOFF_CUDA_CHECK(cuMemFree(devicePtr));
  cache.erase(it);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: release_desc host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __tileoff_launch_nd_f32(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, int32_t extentX, int32_t extentY, int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1 && rank != 2) {
    std::fprintf(stderr,
        "TileOffload error: __tileoff_launch_nd_f32 currently supports "
        "only rank 1 or 2, got rank %d\n",
        rank);
    std::abort();
  }

  TileOffloadValidateCommonLaunchInputs("__tileoff_launch_nd_f32", rank, blockX, blockY,
      blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);
  TileOffloadDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? TileOffloadCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned gridZ = 1;

  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = TileOffloadElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes =
      TileOffloadCheckedMul(elemCount, sizeof(float), "f32 launch byte count");

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  TILEOFF_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dA, a, numBytes));
  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dB, b, numBytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch binary kernel id=%d rank=%d "
        "grid=(%u,%u,%u) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, gridX, gridY, gridZ, blockX, blockY, blockZ, cudaBlockX,
        extentX, extentY, extentZ, numBytes);
  }

  if (rank == 1) {
    TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);

    TileOffloadHiddenTritonArgs hidden;

    void *args[] = {
        &dA,
        &dB,
        &dC,
        &extentX,
        &hidden.hidden0,
        &hidden.hidden1,
    };

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: binary rank1 args: "
          "dA=0x%llx dB=0x%llx dC=0x%llx extentX=%d "
          "hidden0=0x%llx hidden1=0x%llx "
          "args={%p,%p,%p,%p,%p,%p}\n",
          static_cast<unsigned long long>(dA),
          static_cast<unsigned long long>(dB),
          static_cast<unsigned long long>(dC), extentX,
          static_cast<unsigned long long>(hidden.hidden0),
          static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
          args[2], args[3], args[4], args[5]);

      std::fprintf(stderr, "TileOffload: about to cuLaunchKernel rank1\n");
      std::fflush(stderr);
    }

    TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

    TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
        TileOffloadActiveContextState().stream, args, nullptr));
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr, "TileOffload: cuLaunchKernel rank1 returned\n");
      std::fflush(stderr);
    }

  } else {
    TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);

    TileOffloadHiddenTritonArgs hidden;

    void *args[] = {
        &dA,
        &dB,
        &dC,
        &extentX,
        &extentY,
        &hidden.hidden0,
        &hidden.hidden1,
    };

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: binary rank2 args: "
          "dA=0x%llx dB=0x%llx dC=0x%llx "
          "extentX=%d extentY=%d "
          "hidden0=0x%llx hidden1=0x%llx "
          "args={%p,%p,%p,%p,%p,%p,%p}\n",
          static_cast<unsigned long long>(dA),
          static_cast<unsigned long long>(dB),
          static_cast<unsigned long long>(dC), extentX, extentY,
          static_cast<unsigned long long>(hidden.hidden0),
          static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
          args[2], args[3], args[4], args[5], args[6]);

      std::fprintf(stderr, "TileOffload: about to cuLaunchKernel rank2\n");
      std::fflush(stderr);
    }

    TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

    TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
        TileOffloadActiveContextState().stream, args, nullptr));
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr, "TileOffload: cuLaunchKernel rank2 returned\n");
      std::fflush(stderr);
    }
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: waiting for runtime stream\n");
    std::fflush(stderr);
  }

  TileOffloadWaitForRuntimeStream();

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: runtime stream completed\n");
    std::fflush(stderr);
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: about to cuMemcpyDtoH c=%p dC=0x%llx bytes=%zu\n",
        static_cast<void *>(c), static_cast<unsigned long long>(dC), numBytes);
    std::fflush(stderr);
  }

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(c, dC, numBytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: cuMemcpyDtoH returned\n");
    std::fflush(stderr);
  }

  TILEOFF_CUDA_CHECK(cuMemFree(dA));
  TILEOFF_CUDA_CHECK(cuMemFree(dB));
  TILEOFF_CUDA_CHECK(cuMemFree(dC));
}

// -------------------------------------------------------------------------- //
// Public runtime ABI v2: variadic metadata-driven elementwise launches
// -------------------------------------------------------------------------- //

struct TileOffloadPendingArrayV2 {
  void *host = nullptr;
  std::size_t bytes = 0;
  int32_t flags = 0;
  int64_t lower[3] = {1, 1, 1};
  int64_t stride[3] = {1, 0, 0};
  bool bound = false;
};

struct TileOffloadPendingScalarV2 {
  alignas(8) uint64_t storage = 0;
  std::size_t bytes = 0;
  bool bound = false;
};

struct TileOffloadPendingReductionResultV2 {
  void *host = nullptr;
  alignas(8) uint64_t initialStorage = 0;
  std::size_t bytes = 0;
  bool bound = false;
};

namespace {
struct TileOffloadPendingLaunchV2 {
  bool active = false;
  CUcontext context = nullptr;
  int32_t kernelId = -1;
  int32_t rank = 0;
  int32_t block[3] = {1, 1, 1};
  int32_t extent[3] = {1, 1, 1};
  int32_t loopLower[3] = {1, 1, 1};
  std::vector<TileOffloadPendingArrayV2> arrays;
  std::vector<TileOffloadPendingScalarV2> scalars;
  std::vector<TileOffloadPendingReductionResultV2> reductionResults;
  std::vector<TileOffloadDeviceArg> deviceArgs;
  std::vector<CUdeviceptr> devicePointers;
  std::vector<int32_t> parameterValues;
  std::vector<void *> arguments;
};

} // namespace

static thread_local TileOffloadPendingLaunchV2 TileOffloadPendingLaunchStateV2;

static void TileOffloadClearPendingLaunchV2() {
  auto &pending = TileOffloadPendingLaunchStateV2;
  pending.active = false;
  pending.context = nullptr;
  pending.kernelId = -1;
  pending.rank = 0;
  std::fill_n(pending.block, 3, 1);
  std::fill_n(pending.extent, 3, 1);
  std::fill_n(pending.loopLower, 3, 1);
  pending.arrays.clear();
  pending.scalars.clear();
  pending.reductionResults.clear();
  pending.deviceArgs.clear();
  pending.devicePointers.clear();
  pending.parameterValues.clear();
  pending.arguments.clear();
}

extern "C" void __tileoff_begin_launch_v2(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t extentX,
    int32_t extentY, int32_t extentZ, int32_t loopLowerX, int32_t loopLowerY,
    int32_t loopLowerZ, int32_t arrayCount, int32_t scalarCount) {

  // For profiling
  TILEOFF_PROFILE_SCOPE("TileOffload.begin");

  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (TileOffloadPendingLaunchStateV2.active) {
    std::fprintf(stderr,
        "TileOffload error: nested or incomplete v2 launch on the same host "
        "thread\n");
    std::abort();
  }

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  if (!desc || desc->launchAbiVersion != 2) {
    std::fprintf(stderr,
        "TileOffload error: kernel id %d does not use launch ABI v2\n", kernelId);
    std::abort();
  }
  if (rank != desc->rank || rank < 1 || rank > 2 ||
      arrayCount != desc->arrayCount || scalarCount != desc->scalarCount ||
      arrayCount <= 0 || scalarCount < 0) {
    std::fprintf(stderr,
        "TileOffload error: v2 launch ABI count/rank mismatch for kernel id %d\n",
        kernelId);
    std::abort();
  }
  TileOffloadValidateHostLaunchAgainstDesc(
      desc, kernelId, rank, blockX, blockY, blockZ);

  TileOffloadPendingLaunchV2 &pending = TileOffloadPendingLaunchStateV2;
  pending.active = true;
  pending.context = TileOffloadActiveContext;
  pending.kernelId = kernelId;
  pending.rank = rank;
  pending.block[0] = blockX;
  pending.block[1] = blockY;
  pending.block[2] = blockZ;
  pending.extent[0] = extentX;
  pending.extent[1] = extentY;
  pending.extent[2] = extentZ;
  pending.loopLower[0] = loopLowerX;
  pending.loopLower[1] = loopLowerY;
  pending.loopLower[2] = loopLowerZ;
  pending.arrays.resize(static_cast<std::size_t>(arrayCount));
  pending.scalars.resize(static_cast<std::size_t>(scalarCount));
  pending.reductionResults.resize(static_cast<std::size_t>(desc->outputCount));
}

extern "C" void __tileoff_bind_array_v2(int32_t slot, void *host, int64_t bytes,
    int32_t flags, int64_t lowerX, int64_t lowerY, int64_t lowerZ,
    int64_t strideX, int64_t strideY, int64_t strideZ) {

  // For profiling
  TILEOFF_PROFILE_SCOPE("TileOffload.bind_array");

  // Binding state is thread-local; no registry or CUDA state is accessed.
  if (!TileOffloadPendingLaunchStateV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >= TileOffloadPendingLaunchStateV2.arrays.size() ||
      !host || bytes <= 0 || (flags & 3) == 0 || (flags & ~3) != 0) {
    std::fprintf(stderr, "TileOffload error: invalid v2 array binding\n");
    std::abort();
  }

  TileOffloadPendingArrayV2 &array = TileOffloadPendingLaunchStateV2.arrays[slot];
  if (array.bound) {
    std::fprintf(stderr, "TileOffload error: v2 array slot %d bound twice\n", slot);
    std::abort();
  }
  array.host = host;
  array.bytes = static_cast<std::size_t>(bytes);
  array.flags = flags;
  array.lower[0] = lowerX;
  array.lower[1] = lowerY;
  array.lower[2] = lowerZ;
  array.stride[0] = strideX;
  array.stride[1] = strideY;
  array.stride[2] = strideZ;
  array.bound = true;
}

template <typename T> static void TileOffloadBindScalarV2(int32_t slot, T value) {

  // For profiling
  TILEOFF_PROFILE_SCOPE("TileOffload.bind_scalar");

  // Binding state is thread-local; no registry or CUDA state is accessed.
  if (!TileOffloadPendingLaunchStateV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >= TileOffloadPendingLaunchStateV2.scalars.size()) {
    std::fprintf(stderr, "TileOffload error: invalid v2 scalar binding\n");
    std::abort();
  }
  TileOffloadPendingScalarV2 &scalar = TileOffloadPendingLaunchStateV2.scalars[slot];
  if (scalar.bound) {
    std::fprintf(stderr, "TileOffload error: v2 scalar slot %d bound twice\n", slot);
    std::abort();
  }
  static_assert(sizeof(T) <= sizeof(scalar.storage));
  std::memcpy(&scalar.storage, &value, sizeof(T));
  scalar.bytes = sizeof(T);
  scalar.bound = true;
}

#define TILEOFF_DEFINE_SCALAR_BINDER(SUFFIX, TYPE) \
  extern "C" void __tileoff_bind_scalar_##SUFFIX##_v2( \
      int32_t slot, TYPE value) { \
    TileOffloadBindScalarV2<TYPE>(slot, value); \
  }

TILEOFF_DEFINE_SCALAR_BINDER(i8, int8_t)
TILEOFF_DEFINE_SCALAR_BINDER(i16, int16_t)
TILEOFF_DEFINE_SCALAR_BINDER(i32, int32_t)
TILEOFF_DEFINE_SCALAR_BINDER(i64, int64_t)
TILEOFF_DEFINE_SCALAR_BINDER(f32, float)
TILEOFF_DEFINE_SCALAR_BINDER(f64, double)

#undef TILEOFF_DEFINE_SCALAR_BINDER

template <typename T>
static void TileOffloadBindReductionResultAtV2(
    int32_t slot, T *host, T initialValue) {
  // Binding state is thread-local; no registry or CUDA state is accessed.
  if (!TileOffloadPendingLaunchStateV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >=
          TileOffloadPendingLaunchStateV2.reductionResults.size() ||
      !host || TileOffloadPendingLaunchStateV2.reductionResults[slot].bound) {
    std::fprintf(stderr, "TileOffload error: invalid v2 reduction result binding\n");
    std::abort();
  }
  TileOffloadPendingReductionResultV2 &result =
      TileOffloadPendingLaunchStateV2.reductionResults[slot];
  static_assert(sizeof(T) <= sizeof(result.initialStorage));
  result.host = host;
  std::memcpy(&result.initialStorage, &initialValue, sizeof(T));
  result.bytes = sizeof(T);
  result.bound = true;
}

#define TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(SUFFIX, TYPE) \
  extern "C" void __tileoff_bind_reduction_result_##SUFFIX##_v2( \
      TYPE *host, TYPE initialValue) { \
    TileOffloadBindReductionResultAtV2<TYPE>(0, host, initialValue); \
  } \
  extern "C" void __tileoff_bind_reduction_result_##SUFFIX##_at_v2( \
      int32_t slot, TYPE *host, TYPE initialValue) { \
    TileOffloadBindReductionResultAtV2<TYPE>(slot, host, initialValue); \
  }

TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(i8, int8_t)
TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(i16, int16_t)
TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(i32, int32_t)
TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(i64, int64_t)
TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(f32, float)
TILEOFF_DEFINE_REDUCTION_RESULT_BINDER(f64, double)

#undef TILEOFF_DEFINE_REDUCTION_RESULT_BINDER

extern "C" void __tileoff_launch_reduce_f32_v2(
    int32_t, int32_t, int32_t, float *, float *, float *, float, int32_t);
extern "C" void __tileoff_launch_reduce_f64_v2(
    int32_t, int32_t, int32_t, double *, double *, double *, double, int32_t);
extern "C" void __tileoff_launch_reduce_i8_v2(
    int32_t, int32_t, int32_t, int8_t *, int8_t *, int8_t *, int8_t, int32_t);
extern "C" void __tileoff_launch_reduce_i16_v2(int32_t, int32_t, int32_t,
    int16_t *, int16_t *, int16_t *, int16_t, int32_t);
extern "C" void __tileoff_launch_reduce_i32_v2(int32_t, int32_t, int32_t,
    int32_t *, int32_t *, int32_t *, int32_t, int32_t);
extern "C" void __tileoff_launch_reduce_i64_v2(int32_t, int32_t, int32_t,
    int64_t *, int64_t *, int64_t *, int64_t, int32_t);
extern "C" void __tileoff_launch_matmul_f32_v1(int32_t, int32_t, int32_t, int32_t,
    float *, float *, float *, int32_t, int32_t, int32_t);
extern "C" void __tileoff_launch_matmul_f64_v1(int32_t, int32_t, int32_t, int32_t,
    double *, double *, double *, int32_t, int32_t, int32_t);

static CUdeviceptr TileOffloadReserveReductionBuffer(TileOffloadDeviceAllocation &,
    std::size_t, TileOffloadReductionBufferStats &, const char *);
template <typename Real>
static Real TileOffloadReductionIdentity(TileOffloadKernelDesc::ReductionOperator);
template <typename Real>
static Real TileOffloadApplyReduction(TileOffloadKernelDesc::ReductionOperator, Real, Real);
template <typename Real>
static bool TileOffloadEnqueueReductionOnDevice(const TileOffloadKernelDesc *,
    TileOffloadReductionWorkspace &, CUdeviceptr, unsigned, CUdeviceptr *);
template <typename Real>
static bool TileOffloadFinalizeReductionOnDevice(const TileOffloadKernelDesc *,
    TileOffloadReductionWorkspace &, CUdeviceptr, unsigned, Real *);
static int32_t TileOffloadCheckedI32Layout(int64_t, const char *);

template <typename T>
static T TileOffloadPendingReductionInitial(
    const TileOffloadPendingReductionResultV2 &result) {
  T initialValue;
  std::memcpy(&initialValue, &result.initialStorage, sizeof(T));
  return initialValue;
}

template <typename T>
static void TileOffloadCommitReductionTypedV2(
    const TileOffloadKernelDesc *desc, TileOffloadPendingLaunchV2 &pending) {
  TileOffloadPendingReductionResultV2 &pendingResult = pending.reductionResults[0];
  T initialValue = TileOffloadPendingReductionInitial<T>(pendingResult);
  T *result = static_cast<T *>(pendingResult.host);
  if (pending.extent[0] <= 0) {
    *result = initialValue;
    TileOffloadClearPendingLaunchV2();
    return;
  }

  if (desc->backend == "cuda-tile") {
    // The first Tile reduction emitter uses signed i32 layout arithmetic.
    // Include inactive tail lanes in this check since pointer offsets are
    // computed before masked loads. Host initial values never enter a tile.
    if (pending.block[0] < 2 || pending.block[0] > 1024 ||
        (pending.block[0] & (pending.block[0] - 1)) != 0) {
      std::fprintf(stderr, "TileOffload error: invalid CUDA Tile reduction tile\n");
      std::abort();
    }
    int64_t padded = ((int64_t(pending.extent[0]) + pending.block[0] - 1) /
                         pending.block[0]) *
        pending.block[0];
    for (const auto &array : pending.arrays) {
      int64_t first = int64_t(pending.loopLower[0]) -
          TileOffloadCheckedI32Layout(array.lower[0], "reduction array lower bound");
      int64_t last = first + pending.extent[0] - 1;
      int64_t paddedLast = first + padded - 1;
      int64_t stride = array.stride[0];
      // Division checks keep the validation itself free of overflow.
      if (first < 0 || paddedLast > INT32_MAX || stride <= 0 ||
          stride > INT32_MAX || paddedLast > INT32_MAX / stride ||
          uint64_t(last * stride) >= array.bytes / sizeof(T)) {
        std::fprintf(stderr,
            "TileOffload error: CUDA Tile reduction layout is out of bounds or "
            "exceeds i32 indexing; use the Triton backend\n");
        std::abort();
      }
    }
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned gridX = TileOffloadCdiv(
      pending.extent[0], pending.block[0], "v2 reduction grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(desc);
  if (desc->backend != "cuda-tile")
    TileOffloadValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  TileOffloadValidateSupportedHiddenPtrArgCount(pending.kernelId, desc);

  auto &deviceArgs = pending.deviceArgs;
  auto &devicePointers = pending.devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    TileOffloadPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 1) == 0) {
      std::fprintf(stderr,
          "TileOffload error: v2 reduction input is not readable for kernel id "
          "%d\n",
          pending.kernelId);
      std::abort();
    }
    TileOffloadDeviceArg device =
        TileOffloadPrepareArrayBuffer(desc, static_cast<int32_t>(slot), array.host,
            array.bytes, /*read-only reduction input=*/1);
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }

  TileOffloadReductionWorkspace &workspace = TileOffloadGetReductionWorkspace();
  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(T);
  CUdeviceptr dPartials = TileOffloadReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  auto &parameterValues = pending.parameterValues;
  parameterValues.assign(desc->parameters.size(), 0);
  auto &arguments = pending.arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const TileOffloadKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case TileOffloadKernelParameterRole::Read:
      if ((pending.arrays[parameter.arrayIndex].flags & 1) == 0) {
        std::fprintf(stderr,
            "TileOffload error: reduction array flags disagree with kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    case TileOffloadKernelParameterRole::Partials:
      arguments.push_back(&dPartials);
      break;
    case TileOffloadKernelParameterRole::Scalar: {
      TileOffloadPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != desc->scalarParameterBytes[parameter.slot]) {
        std::fprintf(stderr,
            "TileOffload error: reduction scalar type mismatch for kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case TileOffloadKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case TileOffloadKernelParameterRole::ArrayLowerBound:
    case TileOffloadKernelParameterRole::ArrayStride: {
      const TileOffloadPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = TileOffloadCheckedI32Layout(value,
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    default:
      std::fprintf(stderr,
          "TileOffload error: unsupported reduction v2 parameter for kernel id %d "
          "slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  TileOffloadHiddenTritonArgs hidden;
  if (desc->tritonHiddenPtrArgs == 2) {
    arguments.push_back(&hidden.hidden0);
    arguments.push_back(&hidden.hidden1);
  }

  unsigned dynamicSharedBytes;
  if constexpr (std::is_same_v<T, double>) {
    dynamicSharedBytes =
        TileOffloadReductionF64DynamicSharedBytes(desc, pending.block[0]);
  } else if constexpr (std::is_integral_v<T>) {
    dynamicSharedBytes = TileOffloadReductionIntegerDynamicSharedBytes(
        desc, pending.block[0], sizeof(T));
  } else {
    dynamicSharedBytes =
        TileOffloadReductionDynamicSharedBytes(desc, pending.block[0]);
  }
  TileOffloadConfigureDynamicSharedMemory(
      function, pending.kernelId, dynamicSharedBytes);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch reduction v2 kernel id=%d arrays=%zu scalars=%zu "
        "grid=(%u,1,1) extent=%d\n",
        pending.kernelId, pending.arrays.size(), pending.scalars.size(), gridX,
        pending.extent[0]);
  }
  TILEOFF_CUDA_CHECK(cuLaunchKernel(function, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, arguments.data(),
      nullptr));
  ++workspace.primaryLaunches;

  T reducedValue = TileOffloadReductionIdentity<T>(desc->reductionOp);
  if (!TileOffloadFinalizeReductionOnDevice<T>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    TileOffloadWaitForRuntimeStream();
    std::vector<T> partials(gridX);
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(partials.data(), dPartials, partialBytes));
    for (T value : partials)
      reducedValue =
          TileOffloadApplyReduction(desc->reductionOp, reducedValue, value);
  }
  *result = TileOffloadApplyReduction(desc->reductionOp, initialValue, reducedValue);

  for (const TileOffloadDeviceArg &device : deviceArgs)
    TileOffloadReleaseDeviceArg(device);
  TileOffloadClearPendingLaunchV2();
}

template <typename T>
static void TileOffloadCommitMultiReductionTypedV2(
    const TileOffloadKernelDesc *desc, TileOffloadPendingLaunchV2 &pending) {
  if (pending.reductionResults.size() !=
          static_cast<std::size_t>(desc->outputCount) ||
      pending.reductionResults.empty()) {
    std::fprintf(stderr,
        "TileOffload error: multi-reduction result count mismatch for kernel id "
        "%d\n",
        pending.kernelId);
    std::abort();
  }

  for (const TileOffloadPendingReductionResultV2 &result : pending.reductionResults)
    if (!result.bound || result.bytes != sizeof(T)) {
      std::fprintf(stderr,
          "TileOffload error: incomplete multi-reduction result bindings for "
          "kernel id %d\n",
          pending.kernelId);
      std::abort();
    }

  if (pending.extent[0] <= 0 || pending.extent[1] <= 0) {
    for (TileOffloadPendingReductionResultV2 &result : pending.reductionResults)
      *static_cast<T *>(result.host) = TileOffloadPendingReductionInitial<T>(result);
    TileOffloadClearPendingLaunchV2();
    return;
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned gridX = TileOffloadCdiv(pending.extent[0], pending.block[0],
      "v2 multi-reduction grid dimension X");
  unsigned gridY = TileOffloadCdiv(pending.extent[1], pending.block[1],
      "v2 multi-reduction grid dimension Y");
  std::size_t programCountSize =
      TileOffloadCheckedMul(static_cast<std::size_t>(gridX),
          static_cast<std::size_t>(gridY), "multi-reduction program count");
  if (programCountSize >
      static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "TileOffload error: multi-reduction program count exceeds unsigned\n");
    std::abort();
  }
  unsigned programCount = static_cast<unsigned>(programCountSize);

  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(desc);
  TileOffloadValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  TileOffloadValidateSupportedHiddenPtrArgCount(pending.kernelId, desc);

  auto &deviceArgs = pending.deviceArgs;
  auto &devicePointers = pending.devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    TileOffloadPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 1) == 0) {
      std::fprintf(stderr,
          "TileOffload error: multi-reduction input is not readable for kernel id "
          "%d\n",
          pending.kernelId);
      std::abort();
    }
    TileOffloadDeviceArg device =
        TileOffloadPrepareArrayBuffer(desc, static_cast<int32_t>(slot), array.host,
            array.bytes, /*read-only reduction input=*/1);
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }

  TileOffloadReductionWorkspace &workspace = TileOffloadGetReductionWorkspace();
  std::size_t partialElements = TileOffloadCheckedMul(programCountSize,
      pending.reductionResults.size(), "multi-reduction partial elements");
  std::size_t partialBytes = TileOffloadCheckedMul(
      partialElements, sizeof(T), "multi-reduction partial bytes");
  CUdeviceptr dPartials = TileOffloadReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  auto &parameterValues = pending.parameterValues;
  parameterValues.assign(desc->parameters.size(), 0);
  auto &arguments = pending.arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const TileOffloadKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case TileOffloadKernelParameterRole::Read:
      if ((pending.arrays[parameter.arrayIndex].flags & 1) == 0) {
        std::fprintf(stderr,
            "TileOffload error: multi-reduction array flags disagree with kernel "
            "id %d slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    case TileOffloadKernelParameterRole::Partials:
      arguments.push_back(&dPartials);
      break;
    case TileOffloadKernelParameterRole::Scalar: {
      TileOffloadPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != desc->scalarParameterBytes[parameter.slot]) {
        std::fprintf(stderr,
            "TileOffload error: multi-reduction scalar type mismatch for kernel "
            "id %d slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case TileOffloadKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case TileOffloadKernelParameterRole::ExtentY:
      arguments.push_back(&pending.extent[1]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerY:
      arguments.push_back(&pending.loopLower[1]);
      break;
    case TileOffloadKernelParameterRole::ArrayLowerBound:
    case TileOffloadKernelParameterRole::ArrayStride: {
      const TileOffloadPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = TileOffloadCheckedI32Layout(value,
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    default:
      std::fprintf(stderr,
          "TileOffload error: unsupported multi-reduction v2 parameter for kernel "
          "id %d slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  TileOffloadHiddenTritonArgs hidden;
  arguments.push_back(&hidden.hidden0);
  arguments.push_back(&hidden.hidden1);

  std::size_t tileElementsSize =
      TileOffloadCheckedMul(static_cast<std::size_t>(pending.block[0]),
          static_cast<std::size_t>(pending.block[1]),
          "multi-reduction tile elements");
  if (tileElementsSize >
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    std::fprintf(stderr,
        "TileOffload error: multi-reduction tile element count exceeds i32\n");
    std::abort();
  }
  int32_t tileElements = static_cast<int32_t>(tileElementsSize);
  unsigned dynamicSharedBytes;
  if constexpr (std::is_same_v<T, double>) {
    dynamicSharedBytes =
        TileOffloadReductionF64DynamicSharedBytes(desc, tileElements);
  } else if constexpr (std::is_integral_v<T>) {
    dynamicSharedBytes =
        TileOffloadReductionIntegerDynamicSharedBytes(desc, tileElements, sizeof(T));
  } else {
    dynamicSharedBytes = TileOffloadReductionDynamicSharedBytes(desc, tileElements);
  }
  TileOffloadConfigureDynamicSharedMemory(
      function, pending.kernelId, dynamicSharedBytes);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch multi-reduction v2 kernel id=%d arrays=%zu "
        "scalars=%zu outputs=%zu grid=(%u,%u,1) extent=(%d,%d)\n",
        pending.kernelId, pending.arrays.size(), pending.scalars.size(),
        pending.reductionResults.size(), gridX, gridY, pending.extent[0],
        pending.extent[1]);
  }
  TILEOFF_CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, arguments.data(),
      nullptr));
  ++workspace.primaryLaunches;

  const std::size_t resultCount = pending.reductionResults.size();
  std::size_t resultStrideBytes = TileOffloadCheckedMul(
      programCountSize, sizeof(T), "multi-reduction result partial stride");
  std::vector<T> reduced(
      resultCount, TileOffloadReductionIdentity<T>(desc->reductionOp));
  if (desc->reductionStageId >= 0) {
    std::size_t resultBytes = TileOffloadCheckedMul(
        resultCount, sizeof(T), "packed multi-reduction results");
    CUdeviceptr packed = TileOffloadReserveReductionBuffer(
        workspace.results, resultBytes, workspace.resultStats, "results");
    for (std::size_t index = 0; index < resultCount; ++index) {
      CUdeviceptr segment = dPartials + index * resultStrideBytes;
      CUdeviceptr finalValue = 0;
      if (!TileOffloadEnqueueReductionOnDevice<T>(
              desc, workspace, segment, programCount, &finalValue))
        std::abort(); // A nonnegative stage id must resolve or diagnose.
      // Capture the value before the next output reuses the scratch buffer.
      TILEOFF_CUDA_CHECK(cuMemcpyDtoDAsync(packed + index * sizeof(T), finalValue,
          sizeof(T), TileOffloadActiveContextState().stream));
    }
    TileOffloadWaitForRuntimeStream();
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(reduced.data(), packed, resultBytes));
  } else {
    // Keep the original per-output arithmetic order in the host fallback.
    std::size_t count = TileOffloadCheckedMul(
        resultCount, programCountSize, "multi-reduction host partial count");
    std::vector<T> partials(count);
    TileOffloadWaitForRuntimeStream();
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(partials.data(), dPartials,
        TileOffloadCheckedMul(
            count, sizeof(T), "multi-reduction host partial bytes")));
    for (std::size_t index = 0; index < resultCount; ++index)
      for (std::size_t i = 0; i < programCountSize; ++i)
        reduced[index] = TileOffloadApplyReduction(desc->reductionOp, reduced[index],
            partials[index * programCountSize + i]);
  }
  for (std::size_t index = 0; index < resultCount; ++index) {
    TileOffloadPendingReductionResultV2 &binding = pending.reductionResults[index];
    *static_cast<T *>(binding.host) = TileOffloadApplyReduction(desc->reductionOp,
        TileOffloadPendingReductionInitial<T>(binding), reduced[index]);
  }

  for (const TileOffloadDeviceArg &device : deviceArgs)
    TileOffloadReleaseDeviceArg(device);
  TileOffloadClearPendingLaunchV2();
}

static bool TileOffloadTryCommitReductionLaunchV2(
    const TileOffloadKernelDesc *desc, TileOffloadPendingLaunchV2 &pending) {
  if (!desc->isReduction)
    return false;
  if (pending.arrays.empty()) {
    std::fprintf(stderr,
        "TileOffload error: incomplete v2 reduction bindings for kernel id %d\n",
        pending.kernelId);
    std::abort();
  }

  std::string pointerType;
  for (const TileOffloadKernelParameterDesc &parameter : desc->parameters)
    if (parameter.role == TileOffloadKernelParameterRole::Read) {
      pointerType = parameter.type;
      break;
    }
  bool isMultiReduction = desc->kind == "reduction_multi2d";
  if ((!isMultiReduction && pending.reductionResults.size() != 1) ||
      pending.reductionResults.empty()) {
    std::fprintf(stderr,
        "TileOffload error: invalid v2 reduction result count for kernel id %d\n",
        pending.kernelId);
    std::abort();
  }
  std::size_t resultBytes = pending.reductionResults[0].bytes;
  for (const TileOffloadPendingReductionResultV2 &result : pending.reductionResults)
    if (!result.bound || result.bytes != resultBytes) {
      std::fprintf(stderr,
          "TileOffload error: incomplete v2 reduction result bindings for kernel "
          "id %d\n",
          pending.kernelId);
      std::abort();
    }

#define TILEOFF_DISPATCH_REDUCTION(TYPE_NAME, TYPE) \
  if (pointerType == "ptr<" TYPE_NAME ">" && resultBytes == sizeof(TYPE)) { \
    if (isMultiReduction) \
      TileOffloadCommitMultiReductionTypedV2<TYPE>(desc, pending); \
    else \
      TileOffloadCommitReductionTypedV2<TYPE>(desc, pending); \
    return true; \
  }

  TILEOFF_DISPATCH_REDUCTION("i8", int8_t)
  TILEOFF_DISPATCH_REDUCTION("i16", int16_t)
  TILEOFF_DISPATCH_REDUCTION("i32", int32_t)
  TILEOFF_DISPATCH_REDUCTION("i64", int64_t)
  TILEOFF_DISPATCH_REDUCTION("f32", float)
  TILEOFF_DISPATCH_REDUCTION("f64", double)

#undef TILEOFF_DISPATCH_REDUCTION

  std::fprintf(stderr,
      "TileOffload error: v2 reduction result type mismatch for kernel id %d\n",
      pending.kernelId);
  std::abort();
}

static bool TileOffloadTryCommitMatmulLaunchV2(
    const TileOffloadKernelDesc *desc, const TileOffloadPendingLaunchV2 &pending) {
  if (desc->kind != "matmul2d")
    return false;
  // New kernels carry bounds/strides and use the generic checked argument path.
  // Keep the six-argument legacy launcher for previously embedded images.
  if (std::any_of(desc->parameters.begin(), desc->parameters.end(),
          [](const TileOffloadKernelParameterDesc &p) {
            return p.role == TileOffloadKernelParameterRole::LoopLowerZ;
          }))
    return false;

  if (pending.arrays.size() != 3 || (pending.arrays[0].flags & 1) == 0 ||
      (pending.arrays[1].flags & 1) == 0 ||
      (pending.arrays[2].flags & 2) == 0) {
    std::fprintf(stderr,
        "TileOffload error: matmul v2 requires readable A/B and writable C for "
        "kernel id %d\n",
        pending.kernelId);
    std::abort();
  }

  std::string pointerType;
  for (const TileOffloadKernelParameterDesc &parameter : desc->parameters)
    if (parameter.role == TileOffloadKernelParameterRole::Read) {
      pointerType = parameter.type;
      break;
    }

  int32_t kernelId = pending.kernelId;
  int32_t blockX = pending.block[0];
  int32_t blockY = pending.block[1];
  int32_t blockK = pending.block[2];
  void *a = pending.arrays[0].host;
  void *b = pending.arrays[1].host;
  void *c = pending.arrays[2].host;
  int32_t n = pending.extent[0];
  int32_t m = pending.extent[1];
  int32_t k = pending.extent[2];
  TileOffloadClearPendingLaunchV2();

  if (pointerType == "ptr<f32>") {
    __tileoff_launch_matmul_f32_v1(kernelId, blockX, blockY, blockK,
        static_cast<float *>(a), static_cast<float *>(b),
        static_cast<float *>(c), n, m, k);
    return true;
  }
  if (pointerType == "ptr<f64>") {
    __tileoff_launch_matmul_f64_v1(kernelId, blockX, blockY, blockK,
        static_cast<double *>(a), static_cast<double *>(b),
        static_cast<double *>(c), n, m, k);
    return true;
  }

  std::fprintf(stderr,
      "TileOffload error: unsupported matmul v2 element type for kernel id %d\n",
      kernelId);
  std::abort();
}

static int32_t TileOffloadCheckedI32Layout(int64_t value, const char *what) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    std::fprintf(
        stderr, "TileOffload error: %s does not fit the device i32 ABI\n", what);
    std::abort();
  }
  return static_cast<int32_t>(value);
}

extern "C" void __tileoff_commit_launch_v2() {

  // For profiling
  TILEOFF_PROFILE_SCOPE("TileOffload.commit");

  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;

  // For profiling
  TILEOFF_PROFILE_PUSH("TileOffload.context");
  TileOffloadEnsureCurrentContext();

  // For profiling
  TILEOFF_PROFILE_POP();

  TileOffloadPendingLaunchV2 &pending = TileOffloadPendingLaunchStateV2;
  if (!pending.active || pending.context != TileOffloadActiveContext) {
    std::fprintf(stderr,
        "TileOffload error: v2 launch committed without its original CUDA "
        "context\n");
    std::abort();
  }
  for (const TileOffloadPendingArrayV2 &array : pending.arrays)
    if (!array.bound) {
      std::fprintf(stderr, "TileOffload error: incomplete v2 array bindings\n");
      std::abort();
    }
  for (const TileOffloadPendingScalarV2 &scalar : pending.scalars)
    if (!scalar.bound) {
      std::fprintf(stderr, "TileOffload error: incomplete v2 scalar bindings\n");
      std::abort();
    }

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(pending.kernelId);
  int32_t loopSteps[3];
  for (int dim = 0; dim < 3; ++dim) {
    loopSteps[dim] = desc->loopStep[dim];
    int index = desc->loopStepScalarIndex[dim];
    if (index >= 0) {
      if (static_cast<std::size_t>(index) >= pending.scalars.size() ||
          !pending.scalars[index].bound || pending.scalars[index].bytes != 4) {
        std::fprintf(stderr,
            "TileOffload error: missing or invalid runtime loop step binding\n");
        std::abort();
      }
      std::memcpy(&loopSteps[dim], &pending.scalars[index].storage, 4);
    }
    if (!loopSteps[dim]) {
      std::fprintf(stderr, "TileOffload error: runtime loop step must be nonzero\n");
      std::abort();
    }
  }
  bool isReduction = desc->isReduction;
  bool anyReductionResultBound = std::any_of(pending.reductionResults.begin(),
      pending.reductionResults.end(),
      [](const TileOffloadPendingReductionResultV2 &result) { return result.bound; });
  bool allReductionResultsBound = !pending.reductionResults.empty() &&
      std::all_of(pending.reductionResults.begin(),
          pending.reductionResults.end(),
          [](const TileOffloadPendingReductionResultV2 &result) {
            return result.bound;
          });
  bool reductionResultsAgree = isReduction
      ? (pending.reductionResults.size() ==
                static_cast<std::size_t>(desc->outputCount) &&
            allReductionResultsBound)
      : !anyReductionResultBound;
  if (!reductionResultsAgree) {
    std::fprintf(stderr,
        "TileOffload error: v2 reduction result binding disagrees with kernel id "
        "%d\n",
        pending.kernelId);
    std::abort();
  }
  if (desc->isReduction && TileOffloadTryCommitReductionLaunchV2(desc, pending))
    return;
  if (desc->isMatmul && TileOffloadTryCommitMatmulLaunchV2(desc, pending))
    return;

  if (pending.extent[0] <= 0 || (pending.rank >= 2 && pending.extent[1] <= 0)) {
    TileOffloadClearPendingLaunchV2();
    return;
  }

  if (desc->isMatmul) {
    // Each operand's selected rectangle must fit its full bound allocation.
    // K=0 still writes the zero accumulator to C, but does not read A or B.
    const int rows[] = {0, 2, 0};
    const int cols[] = {2, 1, 1};
    for (int operand = 0; operand < 3; ++operand) {
      const auto &parameter = desc->parameters.at(operand);
      const auto &array = pending.arrays.at(parameter.arrayIndex);
      int r = rows[operand], c = cols[operand];
      if (pending.extent[r] <= 0 || pending.extent[c] <= 0)
        continue;
      int64_t dr = int64_t(pending.loopLower[r]) -
          TileOffloadCheckedI32Layout(array.lower[0], "matmul row lower bound");
      int64_t dc = int64_t(pending.loopLower[c]) -
          TileOffloadCheckedI32Layout(array.lower[1], "matmul column lower bound");
      int64_t lastRow = dr + int64_t(pending.extent[r] - 1) * loopSteps[r];
      int64_t lastCol = dc + int64_t(pending.extent[c] - 1) * loopSteps[c];
      int64_t elemBytes = parameter.type == "ptr<f64>" ? 8 : 4;
      // Current data ABI requires contiguous column-major storage.
      bool valid = array.stride[0] == 1 && array.stride[1] > 0 &&
          std::min(dr, lastRow) >= 0 && std::min(dc, lastCol) >= 0 &&
          std::max(dr, lastRow) < array.stride[1] &&
          std::max(dc, lastCol) <
              int64_t(array.bytes / elemBytes / array.stride[1]);
      if (!valid) {
        std::fprintf(stderr,
            "TileOffload error: matmul operand rectangle is outside "
            "its contiguous allocation (kernel %d, operand %d)\n",
            pending.kernelId, operand);
        std::abort();
      }
    }
  }

  // The initial Tile pointwise ABI uses signed i32 flattened coordinates.
  // Check padded tiles too: masked lanes still evaluate their pointer offsets.
  if (desc->backend == "cuda-tile") {
    int64_t padded[2] = {1, 1};
    for (int dim = 0; dim < pending.rank; ++dim) {
      if (dim >= 2 || pending.block[dim] <= 0) {
        std::fprintf(
            stderr, "TileOffload error: invalid CUDA Tile launch geometry\n");
        std::abort();
      }
      padded[dim] = ((int64_t(pending.extent[dim]) + pending.block[dim] - 1) /
                        pending.block[dim]) *
          pending.block[dim];
    }
    if (padded[0] * padded[1] > INT32_MAX) {
      std::fprintf(stderr,
          "TileOffload error: CUDA Tile pointwise indexing exceeds i32 range; "
          "recompile with the Triton backend\n");
      std::abort();
    }
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(desc);
  if (desc->backend != "cuda-tile")
    TileOffloadValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  TileOffloadValidateSupportedHiddenPtrArgCount(pending.kernelId, desc);

  // For profiling
  TILEOFF_PROFILE_PUSH("TileOffload.buffers");

  auto &deviceArgs = pending.deviceArgs;
  auto &devicePointers = pending.devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    TileOffloadPendingArrayV2 &array = pending.arrays[slot];
    TileOffloadDeviceArg device = TileOffloadPrepareArrayBuffer(
        desc, static_cast<int32_t>(slot), array.host, array.bytes, array.flags);
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }
  // For profiling
  TILEOFF_PROFILE_POP();

  // For profiling
  TILEOFF_PROFILE_PUSH("TileOffload.arguments");
  auto &parameterValues = pending.parameterValues;
  parameterValues.assign(desc->parameters.size(), 0);
  auto &arguments = pending.arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const TileOffloadKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case TileOffloadKernelParameterRole::Read:
    case TileOffloadKernelParameterRole::Write:
    case TileOffloadKernelParameterRole::ReadWrite: {
      TileOffloadPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int32_t requiredFlags = parameter.role == TileOffloadKernelParameterRole::Read
          ? 1
          : parameter.role == TileOffloadKernelParameterRole::Write ? 2
                                                              : 3;
      if ((array.flags & requiredFlags) != requiredFlags) {
        std::fprintf(stderr,
            "TileOffload error: v2 array binding flags disagree with kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    }
    case TileOffloadKernelParameterRole::Scalar: {
      TileOffloadPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != desc->scalarParameterBytes[parameter.slot]) {
        std::fprintf(stderr,
            "TileOffload error: v2 scalar binding type mismatch for kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case TileOffloadKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case TileOffloadKernelParameterRole::ExtentY:
      arguments.push_back(&pending.extent[1]);
      break;
    case TileOffloadKernelParameterRole::ExtentZ:
      arguments.push_back(&pending.extent[2]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerY:
      arguments.push_back(&pending.loopLower[1]);
      break;
    case TileOffloadKernelParameterRole::LoopLowerZ:
      arguments.push_back(&pending.loopLower[2]);
      break;
    case TileOffloadKernelParameterRole::ArrayLowerBound:
    case TileOffloadKernelParameterRole::ArrayStride: {
      const TileOffloadPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = TileOffloadCheckedI32Layout(value,
          parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    case TileOffloadKernelParameterRole::Partials:
    case TileOffloadKernelParameterRole::Unknown:
      std::fprintf(stderr,
          "TileOffload error: unsupported parameter in v2 commit for kernel id %d "
          "slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  TileOffloadHiddenTritonArgs hidden;
  if (desc->tritonHiddenPtrArgs == 2) {
    arguments.push_back(&hidden.hidden0);
    arguments.push_back(&hidden.hidden1);
  }

  unsigned gridX =
      TileOffloadCdiv(pending.extent[0], pending.block[0], "v2 grid dimension X");
  unsigned gridY = pending.rank >= 2
      ? TileOffloadCdiv(pending.extent[1], pending.block[1], "v2 grid dimension Y")
      : 1;
  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch %s v2 kernel id=%d arrays=%zu scalars=%zu "
        "grid=(%u,%u,1) extent=(%d,%d) lower=(%d,%d)\n",
        desc->kind.c_str(), pending.kernelId, pending.arrays.size(),
        pending.scalars.size(), gridX, gridY, pending.extent[0],
        pending.extent[1], pending.loopLower[0], pending.loopLower[1]);
  }
  // For profiling
  TILEOFF_PROFILE_POP();

  // For profiling
  TILEOFF_PROFILE_PUSH("TileOffload.launch");
  unsigned sharedBytes = 0;
  if (desc->isMatmul) {
    bool f64 = desc->parameters.front().type == "ptr<f64>";
    sharedBytes = f64 ? TileOffloadMatmulF64DynamicSharedBytes(desc, pending.block[0],
                            pending.block[1], pending.block[2])
                      : TileOffloadMatmulDynamicSharedBytes(desc, pending.block[0],
                            pending.block[1], pending.block[2]);
  }
  TileOffloadConfigureDynamicSharedMemory(function, pending.kernelId, sharedBytes);
  TILEOFF_CUDA_CHECK(
      cuLaunchKernel(function, gridX, gridY, 1, cudaBlockX, 1, 1, sharedBytes,
          TileOffloadActiveContextState().stream, arguments.data(), nullptr));
  TileOffloadCompleteArrayLaunch(std::all_of(
      deviceArgs.begin(), deviceArgs.end(), [](const TileOffloadDeviceArg &arg) {
        return arg.cached && arg.target == TILEOFF_PACK_TARGET_DEVICE;
      }));
  // For profiling
  TILEOFF_PROFILE_POP();

  // For profiling
  TILEOFF_PROFILE_PUSH("TileOffload.completion");
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    TileOffloadPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 2) && deviceArgs[slot].target == TILEOFF_PACK_TARGET_HOST)
      TileOffloadCopyBackWriteBuffer(array.host, deviceArgs[slot], array.bytes);
  }
  for (const TileOffloadDeviceArg &device : deviceArgs)
    TileOffloadReleaseDeviceArg(device);
  // For profiling
  TILEOFF_PROFILE_POP();
  TileOffloadClearPendingLaunchV2();
}

#ifndef TILEOFFLOAD_LAUNCH_V3_H
#define TILEOFFLOAD_LAUNCH_V3_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Host ABI: native pointer layout; compiler and runtime must share this header.
 * This version batches the existing v2 bindings into one public runtime call.
 * Kernel JSON and the device ABI remain v2. */
typedef struct {
  void *host;
  int64_t bytes;
  int32_t flags; /* 1 read, 2 write, 3 read/write */
  int64_t lower[3], stride[3];
} TileOffloadArrayBindingV3;
typedef struct {
  const void *value; /* address of native scalar bytes, copied during call */
  uint32_t bytes; /* 1, 2, 4 or 8; checked against kernel metadata */
} TileOffloadScalarBindingV3;
typedef struct {
  void *host;
  const void *initialValue;
  uint32_t bytes;
} TileOffloadReductionBindingV3;
typedef struct {
  uint32_t version; /* must be 3 */
  uint32_t structBytes; /* sizeof(TileOffloadLaunchV3) */
  int32_t kernelId, rank;
  int32_t block[3], extent[3], loopLower[3];
  int32_t arrayCount, scalarCount, resultCount;
  const TileOffloadArrayBindingV3 *arrays;
  const TileOffloadScalarBindingV3 *scalars;
  const TileOffloadReductionBindingV3 *results;
} TileOffloadLaunchV3;
void __tileoff_launch_v3(const TileOffloadLaunchV3 *launch);
#ifdef __cplusplus
}
#endif
#endif /* TILEOFFLOAD_LAUNCH_V3_H */

extern "C" void __tileoff_launch_v3(const TileOffloadLaunchV3 *launch) {
  TILEOFF_PROFILE_SCOPE("TileOffload.launch_v3");
  if (!launch || launch->version != 3 ||
      launch->structBytes != sizeof(TileOffloadLaunchV3) || launch->arrayCount <= 0 ||
      launch->scalarCount < 0 || launch->resultCount < 0 || !launch->arrays ||
      (launch->scalarCount && !launch->scalars) ||
      (launch->resultCount && !launch->results)) {
    std::fprintf(
        stderr, "TileOffload error: invalid single-call launch v3 descriptor\n");
    std::abort();
  }
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();
  __tileoff_begin_launch_v2(launch->kernelId, launch->rank, launch->block[0],
      launch->block[1], launch->block[2], launch->extent[0], launch->extent[1],
      launch->extent[2], launch->loopLower[0], launch->loopLower[1],
      launch->loopLower[2], launch->arrayCount, launch->scalarCount);
  for (int32_t i = 0; i < launch->arrayCount; ++i) {
    const auto &a = launch->arrays[i];
    __tileoff_bind_array_v2(i, a.host, a.bytes, a.flags, a.lower[0], a.lower[1],
        a.lower[2], a.stride[0], a.stride[1], a.stride[2]);
  }
  for (int32_t i = 0; i < launch->scalarCount; ++i) {
    const auto &a = launch->scalars[i];
    if (!a.value ||
        (a.bytes != 1 && a.bytes != 2 && a.bytes != 4 && a.bytes != 8)) {
      std::fprintf(stderr, "TileOffload error: invalid v3 scalar binding\n");
      std::abort();
    }
    auto &binding = TileOffloadPendingLaunchStateV2.scalars[i];
    std::memcpy(&binding.storage, a.value, a.bytes);
    binding.bytes = a.bytes;
    binding.bound = true;
  }
  if (launch->resultCount &&
      static_cast<std::size_t>(launch->resultCount) !=
          TileOffloadPendingLaunchStateV2.reductionResults.size()) {
    std::fprintf(stderr, "TileOffload error: invalid v3 reduction result count\n");
    std::abort();
  }
  for (int32_t i = 0; i < launch->resultCount; ++i) {
    const auto &a = launch->results[i];
    if (!a.host || !a.initialValue ||
        (a.bytes != 1 && a.bytes != 2 && a.bytes != 4 && a.bytes != 8)) {
      std::fprintf(stderr, "TileOffload error: invalid v3 reduction binding\n");
      std::abort();
    }
    auto &binding = TileOffloadPendingLaunchStateV2.reductionResults[i];
    binding.host = a.host;
    std::memcpy(&binding.initialStorage, a.initialValue, a.bytes);
    binding.bytes = a.bytes;
    binding.bound = true;
  }
  __tileoff_commit_launch_v2();
}

// -------------------------------------------------------------------------- //
// Public runtime ABI: 1-scalar f32 SAXPY-style kernels
// -------------------------------------------------------------------------- //
//
// Current SAXPY TTIR signature:
//
//   (%a: ptr<f32>, %b: ptr<f32>, %c: ptr<f32>, %alpha: f32, %n: i32)
//

extern "C" void __tileoff_launch_nd_f32_s1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, float scalar0, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
        "TileOffload error: __tileoff_launch_nd_f32_s1 currently supports "
        "only rank 1, got rank %d\n",
        rank);
    std::abort();
  }

  TileOffloadValidateCommonLaunchInputs("__tileoff_launch_nd_f32_s1", rank, blockX,
      blockY, blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes =
      TileOffloadCheckedMul(numElems, sizeof(float), "f32 launch byte count");

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  TILEOFF_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dA, a, numBytes));
  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dB, b, numBytes));

  TileOffloadHiddenTritonArgs hidden;

  void *args[] = {
      &dA,
      &dB,
      &dC,
      &scalar0,
      &extentX,
      &hidden.hidden0,
      &hidden.hidden1,
  };

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch SAXPY kernel id=%d rank=%d "
        "grid=(%u,1,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
        "scalar0=%f bytes=%zu\n",
        kernelId, rank, gridX, blockX, blockY, blockZ, cudaBlockX, extentX,
        extentY, extentZ, static_cast<double>(scalar0), numBytes);
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: SAXPY args: "
        "dA=0x%llx dB=0x%llx dC=0x%llx "
        "scalar0=%f extentX=%d "
        "hidden0=0x%llx hidden1=0x%llx "
        "args={%p,%p,%p,%p,%p,%p,%p}\n",
        static_cast<unsigned long long>(dA),
        static_cast<unsigned long long>(dB),
        static_cast<unsigned long long>(dC), static_cast<double>(scalar0),
        extentX, static_cast<unsigned long long>(hidden.hidden0),
        static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
        args[2], args[3], args[4], args[5], args[6]);

    std::fprintf(stderr, "TileOffload: about to cuLaunchKernel SAXPY\n");
    std::fflush(stderr);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
      TileOffloadActiveContextState().stream, args, nullptr));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: cuLaunchKernel SAXPY returned\n");
    std::fflush(stderr);
  }

  TileOffloadWaitForRuntimeStream();

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(c, dC, numBytes));

  TILEOFF_CUDA_CHECK(cuMemFree(dA));
  TILEOFF_CUDA_CHECK(cuMemFree(dB));
  TILEOFF_CUDA_CHECK(cuMemFree(dC));
}

extern "C" void __tileoff_launch_nd_f32_s2(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, float scalar0, float scalar1, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
        "TileOffload error: __tileoff_launch_nd_f32_s2 currently supports "
        "only rank 1, got rank %d\n",
        rank);
    std::abort();
  }

  TileOffloadValidateCommonLaunchInputs("__tileoff_launch_nd_f32_s2", rank, blockX,
      blockY, blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes =
      TileOffloadCheckedMul(numElems, sizeof(float), "f32 launch byte count");

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  TILEOFF_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  TILEOFF_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dA, a, numBytes));
  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(dB, b, numBytes));

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);

  TileOffloadHiddenTritonArgs hidden;

  void *args[] = {
      &dA,
      &dB,
      &dC,
      &scalar0,
      &scalar1,
      &extentX,
      &hidden.hidden0,
      &hidden.hidden1,
  };

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch expr/s2 kernel id=%d rank=%d "
        "grid=(%u,1,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
        "scalar0=%f scalar1=%f bytes=%zu\n",
        kernelId, rank, gridX, blockX, blockY, blockZ, cudaBlockX, extentX,
        extentY, extentZ, static_cast<double>(scalar0),
        static_cast<double>(scalar1), numBytes);

    std::fprintf(stderr,
        "TileOffload: s2 args: "
        "dA=0x%llx dB=0x%llx dC=0x%llx "
        "scalar0=%f scalar1=%f extentX=%d "
        "hidden0=0x%llx hidden1=0x%llx "
        "args={%p,%p,%p,%p,%p,%p,%p,%p}\n",
        static_cast<unsigned long long>(dA),
        static_cast<unsigned long long>(dB),
        static_cast<unsigned long long>(dC), static_cast<double>(scalar0),
        static_cast<double>(scalar1), extentX,
        static_cast<unsigned long long>(hidden.hidden0),
        static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
        args[2], args[3], args[4], args[5], args[6], args[7]);

    std::fprintf(stderr, "TileOffload: about to cuLaunchKernel s2\n");
    std::fflush(stderr);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
      TileOffloadActiveContextState().stream, args, nullptr));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: cuLaunchKernel s2 returned\n");
    std::fflush(stderr);
  }

  TileOffloadWaitForRuntimeStream();

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(c, dC, numBytes));

  TILEOFF_CUDA_CHECK(cuMemFree(dA));
  TILEOFF_CUDA_CHECK(cuMemFree(dB));
  TILEOFF_CUDA_CHECK(cuMemFree(dC));
}

// TileOffload generic f32 launch ABI v1.
//
// This ABI intentionally supports only the compiler subset currently emitted:
//
//   - rank 1 or rank 2
//   - f32 arrays
//   - exactly two read arrays
//   - one write array
//   - zero to three f32 scalar captures
//   - contiguous storage
//   - Triton/NVVM PTX with exactly two hidden pointer arguments
//
// The runtime validates JSON schema version and hidden-argument count so that
// compiler/runtime drift fails explicitly rather than launching with a wrong
// CUDA argument layout.
extern "C" void __tileoff_launch_f32_v1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays,
    int32_t numScalars, float *read0, float *read1, float *read2, float *write,
    float scalar0, float scalar1, float scalar2, int32_t extentX,
    int32_t extentY, int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "TileOffload error: __tileoff_launch_f32_v1 requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "TileOffload error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "TileOffload error: unsupported rank %d in __tileoff_launch_f32_v1\n", rank);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid tile/block shape (%d,%d,%d) in "
        "__tileoff_launch_f32_v1\n",
        blockX, blockY, blockZ);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "TileOffload error: null required pointer in __tileoff_launch_f32_v1: "
        "read0=%p write=%p\n",
        static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(
        stderr, "TileOffload error: null read1 pointer in __tileoff_launch_f32_v1\n");
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(
        stderr, "TileOffload error: null read2 pointer in __tileoff_launch_f32_v1\n");
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? TileOffloadCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = TileOffloadElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes =
      TileOffloadCheckedMul(elemCount, sizeof(float), "f32 launch byte count");

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for generic f32 kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "TileOffload error: generic f32 launcher called for matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  // Current JSON parameter order is:
  //   slot 0 = read0
  //   slot 1 = read1
  //   slot 2 = read2 if present, otherwise write for current kernels
  //   slot numReadArrays = write
  //
  // For current TileOffload kernels numReadArrays is normally 2, so write slot is 2.
  int32_t read0Slot = 0;
  int32_t read1Slot = 1;
  int32_t read2Slot = 2;
  int32_t writeSlot = numReadArrays;

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, read0Slot, read0);

  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, read1Slot, read1)
      : TILEOFF_PACK_TARGET_HOST;

  int32_t read2Target = numReadArrays >= 3
      ? TileOffloadEffectivePackTargetForSlot(desc, read2Slot, read2)
      : TILEOFF_PACK_TARGET_HOST;

  int32_t writeTarget =
      TileOffloadEffectiveWriteTargetForSlot(desc, writeSlot, write);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: pack targets for kernel id %d: "
        "read0(slot %d)=%s/%s read1(slot %d)=%s/%s "
        "read2(slot %d)=%s/%s write(slot %d)=%s/%s\n",
        kernelId, read0Slot, TileOffloadPackTargetName(read0Target),
        TileOffloadPackTargetSourceName(desc, read0Slot, read0), read1Slot,
        TileOffloadPackTargetName(read1Target),
        numReadArrays >= 2 ? TileOffloadPackTargetSourceName(desc, read1Slot, read1)
                           : "unused",
        read2Slot, TileOffloadPackTargetName(read2Target),
        numReadArrays >= 3 ? TileOffloadPackTargetSourceName(desc, read2Slot, read2)
                           : "unused",
        writeSlot, TileOffloadPackTargetName(writeTarget),
        TileOffloadPackTargetSourceName(desc, writeSlot, write));
  }

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, numBytes, read0Target, read0Slot);

  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, numBytes, read1Target, read1Slot);

  TileOffloadDeviceArg read2Dev;
  if (numReadArrays >= 3)
    read2Dev = TileOffloadPrepareReadBuffer(read2, numBytes, read2Target, read2Slot);

  TileOffloadDeviceArg writeDev =
      TileOffloadPrepareWriteBuffer(write, numBytes, writeTarget, writeSlot);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  CUdeviceptr dRead2 = read2Dev.ptr;
  CUdeviceptr dWrite = writeDev.ptr;

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[16];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  if (numReadArrays >= 3)
    args[argCount++] = &dRead2;

  args[argCount++] = &dWrite;

  if (numScalars >= 1)
    args[argCount++] = &scalar0;

  if (numScalars >= 2)
    args[argCount++] = &scalar1;

  if (numScalars >= 3)
    args[argCount++] = &scalar2;

  args[argCount++] = &extentX;

  if (rank >= 2)
    args[argCount++] = &extentY;

  if (rank >= 3)
    args[argCount++] = &extentZ;

  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  if (argCount > 16) {
    std::fprintf(stderr,
        "TileOffload error: internal runtime argument buffer overflow "
        "for kernel id %d; argCount=%d\n",
        kernelId, argCount);
    std::abort();
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch generic f32 kernel id=%d rank=%d "
        "reads=%d scalars=%d grid=(%u,%u,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, numReadArrays, numScalars, gridX, gridY, blockX, blockY,
        blockZ, cudaBlockX, extentX, extentY, extentZ, numBytes);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      TileOffloadActiveContextState().stream, args, nullptr));

  TileOffloadWaitForRuntimeStream();

  if (writeDev.target == TILEOFF_PACK_TARGET_HOST) {
    TileOffloadCopyBackWriteBuffer(write, writeDev, numBytes);
  } else {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipped automatic copy-back for write slot %d "
          "because target=device; use !$tileoff update host(...) to copy back\n",
          writeDev.slot);
    }
  }

  TileOffloadReleaseDeviceArg(read0Dev);

  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);

  if (numReadArrays >= 3)
    TileOffloadReleaseDeviceArg(read2Dev);

  TileOffloadReleaseDeviceArg(writeDev);
}

// TileOffload generic f64 launch ABI v1.
//
// This ABI intentionally supports only the compiler subset currently emitted:
//
//   - rank 1 or rank 2
//   - f32 arrays
//   - exactly two read arrays
//   - one write array
//   - zero to three f64 scalar captures
//   - contiguous storage
//   - Triton/NVVM PTX with exactly two hidden pointer arguments
//
// The runtime validates JSON schema version and hidden-argument count so that
// compiler/runtime drift fails explicitly rather than launching with a wrong
// CUDA argument layout.
extern "C" void __tileoff_launch_f64_v1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays,
    int32_t numScalars, double *read0, double *read1, double *read2,
    double *write, double scalar0, double scalar1, double scalar2,
    int32_t extentX, int32_t extentY, int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "TileOffload error: __tileoff_launch_f64_v1 requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "TileOffload error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "TileOffload error: unsupported rank %d in __tileoff_launch_f64_v1\n", rank);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid tile/block shape (%d,%d,%d) in "
        "__tileoff_launch_f64_v1\n",
        blockX, blockY, blockZ);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "TileOffload error: null required pointer in __tileoff_launch_f64_v1: "
        "read0=%p write=%p\n",
        static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(
        stderr, "TileOffload error: null read1 pointer in __tileoff_launch_f64_v1\n");
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(
        stderr, "TileOffload error: null read2 pointer in __tileoff_launch_f64_v1\n");
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? TileOffloadCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = TileOffloadElementCount(rank, extentX, extentY, extentZ);
  std::size_t numBytes =
      TileOffloadCheckedMul(elemCount, sizeof(double), "f64 launch byte count");

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for generic f64 kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "TileOffload error: generic f64 launcher called for matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  int32_t read0Slot = 0;
  int32_t read1Slot = 1;
  int32_t read2Slot = 2;
  int32_t writeSlot = numReadArrays;

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, read0Slot, read0);

  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, read1Slot, read1)
      : TILEOFF_PACK_TARGET_HOST;

  int32_t read2Target = numReadArrays >= 3
      ? TileOffloadEffectivePackTargetForSlot(desc, read2Slot, read2)
      : TILEOFF_PACK_TARGET_HOST;

  int32_t writeTarget =
      TileOffloadEffectiveWriteTargetForSlot(desc, writeSlot, write);

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, numBytes, read0Target, read0Slot);

  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, numBytes, read1Target, read1Slot);

  TileOffloadDeviceArg read2Dev;
  if (numReadArrays >= 3)
    read2Dev = TileOffloadPrepareReadBuffer(read2, numBytes, read2Target, read2Slot);

  TileOffloadDeviceArg writeDev =
      TileOffloadPrepareWriteBuffer(write, numBytes, writeTarget, writeSlot);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  CUdeviceptr dRead2 = read2Dev.ptr;
  CUdeviceptr dWrite = writeDev.ptr;

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[16];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  if (numReadArrays >= 3)
    args[argCount++] = &dRead2;

  args[argCount++] = &dWrite;

  if (numScalars >= 1)
    args[argCount++] = &scalar0;

  if (numScalars >= 2)
    args[argCount++] = &scalar1;

  if (numScalars >= 3)
    args[argCount++] = &scalar2;

  args[argCount++] = &extentX;

  if (rank >= 2)
    args[argCount++] = &extentY;

  if (rank >= 3)
    args[argCount++] = &extentZ;

  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  if (argCount > 16) {
    std::fprintf(stderr,
        "TileOffload error: internal runtime argument buffer overflow "
        "for kernel id %d; argCount=%d\n",
        kernelId, argCount);
    std::abort();
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch generic f64 kernel id=%d rank=%d "
        "reads=%d scalars=%d grid=(%u,%u,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, numReadArrays, numScalars, gridX, gridY, blockX, blockY,
        blockZ, cudaBlockX, extentX, extentY, extentZ, numBytes);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      TileOffloadActiveContextState().stream, args, nullptr));

  TileOffloadWaitForRuntimeStream();

  if (writeDev.target == TILEOFF_PACK_TARGET_HOST) {
    TileOffloadCopyBackWriteBuffer(write, writeDev, numBytes);
  } else {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipped automatic copy-back for write slot %d "
          "because target=device; use !$tileoff update host(...) to copy back\n",
          writeDev.slot);
    }
  }

  TileOffloadReleaseDeviceArg(read0Dev);

  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);

  if (numReadArrays >= 3)
    TileOffloadReleaseDeviceArg(read2Dev);

  TileOffloadReleaseDeviceArg(writeDev);
}

template <typename Integer>
static void TileOffloadLaunchIntegerV1(const char *abiName, const char *typeName,
    int32_t kernelId, int32_t rank, int32_t blockX, int32_t blockY,
    int32_t blockZ, int32_t numReadArrays, int32_t numScalars, Integer *read0,
    Integer *read1, Integer *read2, Integer *write, Integer scalar0,
    Integer scalar1, Integer scalar2, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "TileOffload error: %s requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        abiName, numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "TileOffload error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(
        stderr, "TileOffload error: unsupported rank %d in %s\n", rank, abiName);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid tile/block shape (%d,%d,%d) in %s\n", blockX,
        blockY, blockZ, abiName);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "TileOffload error: null required pointer in %s: read0=%p write=%p\n",
        abiName, static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(stderr, "TileOffload error: null read1 pointer in %s\n", abiName);
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(stderr, "TileOffload error: null read2 pointer in %s\n", abiName);
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);
  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? TileOffloadCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = TileOffloadElementCount(rank, extentX, extentY, extentZ);
  std::size_t numBytes =
      TileOffloadCheckedMul(elemCount, sizeof(Integer), "integer launch byte count");

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for generic %s kernel id %d\n",
        typeName, kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "TileOffload error: generic %s launcher called for matmul kernel id %d\n",
        typeName, kernelId);
    std::abort();
  }

  int32_t read0Slot = 0;
  int32_t read1Slot = 1;
  int32_t read2Slot = 2;
  int32_t writeSlot = numReadArrays;

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, read0Slot, read0);
  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, read1Slot, read1)
      : TILEOFF_PACK_TARGET_HOST;
  int32_t read2Target = numReadArrays >= 3
      ? TileOffloadEffectivePackTargetForSlot(desc, read2Slot, read2)
      : TILEOFF_PACK_TARGET_HOST;
  int32_t writeTarget =
      TileOffloadEffectiveWriteTargetForSlot(desc, writeSlot, write);

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, numBytes, read0Target, read0Slot);
  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, numBytes, read1Target, read1Slot);
  TileOffloadDeviceArg read2Dev;
  if (numReadArrays >= 3)
    read2Dev = TileOffloadPrepareReadBuffer(read2, numBytes, read2Target, read2Slot);
  TileOffloadDeviceArg writeDev =
      TileOffloadPrepareWriteBuffer(write, numBytes, writeTarget, writeSlot);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  CUdeviceptr dRead2 = read2Dev.ptr;
  CUdeviceptr dWrite = writeDev.ptr;

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[16];
  int argCount = 0;
  args[argCount++] = &dRead0;
  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;
  if (numReadArrays >= 3)
    args[argCount++] = &dRead2;
  args[argCount++] = &dWrite;
  if (numScalars >= 1)
    args[argCount++] = &scalar0;
  if (numScalars >= 2)
    args[argCount++] = &scalar1;
  if (numScalars >= 3)
    args[argCount++] = &scalar2;
  args[argCount++] = &extentX;
  if (rank >= 2)
    args[argCount++] = &extentY;
  if (rank >= 3)
    args[argCount++] = &extentZ;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  if (argCount > 16) {
    std::fprintf(stderr,
        "TileOffload error: internal runtime argument buffer overflow "
        "for kernel id %d; argCount=%d\n",
        kernelId, argCount);
    std::abort();
  }

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch generic %s kernel id=%d rank=%d reads=%d scalars=%d "
        "grid=(%u,%u,1) tile=(%d,%d,%d) cuda_block=(%u,1,1) "
        "extent=(%d,%d,%d) bytes=%zu\n",
        typeName, kernelId, rank, numReadArrays, numScalars, gridX, gridY,
        blockX, blockY, blockZ, cudaBlockX, extentX, extentY, extentZ,
        numBytes);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);
  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      TileOffloadActiveContextState().stream, args, nullptr));
  TileOffloadWaitForRuntimeStream();

  if (writeDev.target == TILEOFF_PACK_TARGET_HOST) {
    TileOffloadCopyBackWriteBuffer(write, writeDev, numBytes);
  } else if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: skipped automatic copy-back for write slot %d "
        "because target=device; use !$tileoff update host(...) to copy back\n",
        writeDev.slot);
  }

  TileOffloadReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);
  if (numReadArrays >= 3)
    TileOffloadReleaseDeviceArg(read2Dev);
  TileOffloadReleaseDeviceArg(writeDev);
}

#define TILEOFF_DEFINE_INTEGER_LAUNCH(BITS, TYPE) \
  extern "C" void __tileoff_launch_i##BITS##_v1(int32_t kernelId, int32_t rank, \
      int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays, \
      int32_t numScalars, TYPE *read0, TYPE *read1, TYPE *read2, TYPE *write, \
      TYPE scalar0, TYPE scalar1, TYPE scalar2, int32_t extentX, \
      int32_t extentY, int32_t extentZ) { \
    TileOffloadLaunchIntegerV1<TYPE>("__tileoff_launch_i" #BITS "_v1", "i" #BITS, \
        kernelId, rank, blockX, blockY, blockZ, numReadArrays, numScalars, \
        read0, read1, read2, write, scalar0, scalar1, scalar2, extentX, \
        extentY, extentZ); \
  }

TILEOFF_DEFINE_INTEGER_LAUNCH(8, int8_t)
TILEOFF_DEFINE_INTEGER_LAUNCH(16, int16_t)
TILEOFF_DEFINE_INTEGER_LAUNCH(32, int32_t)
TILEOFF_DEFINE_INTEGER_LAUNCH(64, int64_t)

#undef TILEOFF_DEFINE_INTEGER_LAUNCH

extern "C" void __tileoff_launch_matmul_f32_v1(int32_t kernelId, int32_t blockX,
    int32_t blockY, int32_t blockK, float *a, float *b, float *c, int32_t n,
    int32_t m, int32_t k) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, 2, blockX, blockY, blockK);

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "TileOffload error: null host pointer in __tileoff_launch_matmul_f32_v1: "
        "a=%p b=%p c=%p\n",
        static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c));
    std::abort();
  }

  if (n <= 0 || m <= 0 || k <= 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipping matmul with non-positive extent n=%d m=%d k=%d\n", n,
          m, k);
    }
    return;
  }

  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr, "TileOffload error: invalid matmul tile shape (%d,%d,%d)\n",
        blockX, blockY, blockK);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  TileOffloadDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = TileOffloadCdiv(n, blockX, "grid dimension X");
  unsigned gridY = TileOffloadCdiv(m, blockY, "grid dimension Y");
  unsigned gridZ = 1;

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: matmul grid debug: "
        "n=%d m=%d blockX=%d blockY=%d "
        "gridX=%u gridY=%u gridZ=%u\n",
        n, m, blockX, blockY, gridX, gridY, gridZ);
  }

  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for matmul kernel id %d\n", kernelId);
    std::abort();
  }

  if (desc->kind != "matmul2d") {
    std::fprintf(stderr,
        "TileOffload error: matmul launcher called for kernel id %d "
        "but JSON kind is '%s'\n",
        kernelId, desc->kind.c_str());
    std::abort();
  }

  unsigned dynamicSharedBytes =
      TileOffloadMatmulDynamicSharedBytes(desc, blockX, blockY, blockK);

  std::size_t bytesA =
      TileOffloadCheckedBytes2D(n, k, sizeof(float), "matmul A bytes");
  std::size_t bytesB =
      TileOffloadCheckedBytes2D(k, m, sizeof(float), "matmul B bytes");
  std::size_t bytesC =
      TileOffloadCheckedBytes2D(n, m, sizeof(float), "matmul C bytes");

  int32_t aTarget = TileOffloadEffectivePackTargetForSlot(desc, 0, a);
  int32_t bTarget = TileOffloadEffectivePackTargetForSlot(desc, 1, b);
  int32_t cTarget = TileOffloadEffectiveWriteTargetForSlot(desc, 2, c);

  TileOffloadDeviceArg aDev = TileOffloadPrepareReadBuffer(a, bytesA, aTarget, 0);
  TileOffloadDeviceArg bDev = TileOffloadPrepareReadBuffer(b, bytesB, bTarget, 1);
  TileOffloadDeviceArg cDev = TileOffloadPrepareWriteBuffer(c, bytesC, cTarget, 2);

  CUdeviceptr dA = aDev.ptr;
  CUdeviceptr dB = bDev.ptr;
  CUdeviceptr dC = cDev.ptr;

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[] = {
      &dA,
      &dB,
      &dC,
      &n,
      &m,
      &k,
      &hidden.hidden0,
      &hidden.hidden1,
  };

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch matmul kernel id=%d "
        "grid=(%u,%u,%u) grid_policy=cdiv tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) dynamic_shared_bytes=%u "
        "n=%d m=%d k=%d "
        "bytesA=%zu bytesB=%zu bytesC=%zu "
        "targets=(%s,%s,%s)\n",
        kernelId, gridX, gridY, gridZ, blockX, blockY, blockK, cudaBlockX,
        dynamicSharedBytes, n, m, k, bytesA, bytesB, bytesC,
        TileOffloadPackTargetName(aTarget), TileOffloadPackTargetName(bTarget),
        TileOffloadPackTargetName(cTarget));

    std::fprintf(stderr,
        "TileOffload: matmul args: "
        "dA=0x%llx dB=0x%llx dC=0x%llx "
        "n=%d m=%d k=%d hidden0=0x%llx hidden1=0x%llx "
        "args={%p,%p,%p,%p,%p,%p,%p,%p}\n",
        static_cast<unsigned long long>(dA),
        static_cast<unsigned long long>(dB),
        static_cast<unsigned long long>(dC), n, m, k,
        static_cast<unsigned long long>(hidden.hidden0),
        static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
        args[2], args[3], args[4], args[5], args[6], args[7]);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);
  TileOffloadConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, gridZ, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, args, nullptr));

  TileOffloadCompleteArrayLaunch(aDev.cached && bDev.cached && cDev.cached &&
      aDev.target == TILEOFF_PACK_TARGET_DEVICE &&
      bDev.target == TILEOFF_PACK_TARGET_DEVICE &&
      cDev.target == TILEOFF_PACK_TARGET_DEVICE);

  if (cDev.target == TILEOFF_PACK_TARGET_HOST) {
    TileOffloadCopyBackWriteBuffer(c, cDev, bytesC);
  } else {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipped automatic copy-back for matmul write slot %d "
          "because target=device; use !$tileoff update host(...) to copy back\n",
          cDev.slot);
    }
  }

  TileOffloadReleaseDeviceArg(aDev);
  TileOffloadReleaseDeviceArg(bDev);
  TileOffloadReleaseDeviceArg(cDev);
}

extern "C" void __tileoff_launch_matmul_f64_v1(int32_t kernelId, int32_t blockX,
    int32_t blockY, int32_t blockK, double *a, double *b, double *c, int32_t n,
    int32_t m, int32_t k) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  TileOffloadValidateHostLaunchAgainstDesc(kernelId, 2, blockX, blockY, blockK);

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "TileOffload error: null host pointer in __tileoff_launch_matmul_f64_v1: "
        "a=%p b=%p c=%p\n",
        static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c));
    std::abort();
  }

  if (n <= 0 || m <= 0 || k <= 0) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipping f64 matmul with non-positive extent "
          "n=%d m=%d k=%d\n",
          n, m, k);
    }
    return;
  }

  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr,
        "TileOffload error: invalid f64 matmul tile shape (%d,%d,%d)\n", blockX,
        blockY, blockK);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  TileOffloadDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = TileOffloadCdiv(n, blockX, "grid dimension X");
  unsigned gridY = TileOffloadCdiv(m, blockY, "grid dimension Y");
  unsigned gridZ = 1;

  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for f64 matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind != "matmul2d") {
    std::fprintf(stderr,
        "TileOffload error: f64 matmul launcher called for kernel id %d "
        "but JSON kind is '%s'\n",
        kernelId, desc->kind.c_str());
    std::abort();
  }

  std::size_t bytesA =
      TileOffloadCheckedBytes2D(n, k, sizeof(double), "f64 matmul A bytes");
  std::size_t bytesB =
      TileOffloadCheckedBytes2D(k, m, sizeof(double), "f64 matmul B bytes");
  std::size_t bytesC =
      TileOffloadCheckedBytes2D(n, m, sizeof(double), "f64 matmul C bytes");

  int32_t aTarget = TileOffloadEffectivePackTargetForSlot(desc, 0, a);
  int32_t bTarget = TileOffloadEffectivePackTargetForSlot(desc, 1, b);
  int32_t cTarget = TileOffloadEffectiveWriteTargetForSlot(desc, 2, c);

  TileOffloadDeviceArg aDev = TileOffloadPrepareReadBuffer(a, bytesA, aTarget, 0);
  TileOffloadDeviceArg bDev = TileOffloadPrepareReadBuffer(b, bytesB, bTarget, 1);
  TileOffloadDeviceArg cDev = TileOffloadPrepareWriteBuffer(c, bytesC, cTarget, 2);

  CUdeviceptr dA = aDev.ptr;
  CUdeviceptr dB = bDev.ptr;
  CUdeviceptr dC = cDev.ptr;

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[] = {
      &dA,
      &dB,
      &dC,
      &n,
      &m,
      &k,
      &hidden.hidden0,
      &hidden.hidden1,
  };

  unsigned dynamicSharedBytes =
      TileOffloadMatmulF64DynamicSharedBytes(desc, blockX, blockY, blockK);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: launch f64 matmul kernel id=%d "
        "grid=(%u,%u,%u) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) dynamic_shared_bytes=%u "
        "n=%d m=%d k=%d "
        "bytesA=%zu bytesB=%zu bytesC=%zu "
        "targets=(%s,%s,%s)\n",
        kernelId, gridX, gridY, gridZ, blockX, blockY, blockK, cudaBlockX,
        dynamicSharedBytes, n, m, k, bytesA, bytesB, bytesC,
        TileOffloadPackTargetName(aTarget), TileOffloadPackTargetName(bTarget),
        TileOffloadPackTargetName(cTarget));

    std::fprintf(stderr,
        "TileOffload: f64 matmul args: "
        "dA=0x%llx dB=0x%llx dC=0x%llx "
        "n=%d m=%d k=%d hidden0=0x%llx hidden1=0x%llx "
        "args={%p,%p,%p,%p,%p,%p,%p,%p}\n",
        static_cast<unsigned long long>(dA),
        static_cast<unsigned long long>(dB),
        static_cast<unsigned long long>(dC), n, m, k,
        static_cast<unsigned long long>(hidden.hidden0),
        static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
        args[2], args[3], args[4], args[5], args[6], args[7]);
  }

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);
  TileOffloadConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, gridZ, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, args, nullptr));

  TileOffloadCompleteArrayLaunch(aDev.cached && bDev.cached && cDev.cached &&
      aDev.target == TILEOFF_PACK_TARGET_DEVICE &&
      bDev.target == TILEOFF_PACK_TARGET_DEVICE &&
      cDev.target == TILEOFF_PACK_TARGET_DEVICE);

  if (cDev.target == TILEOFF_PACK_TARGET_HOST) {
    TileOffloadCopyBackWriteBuffer(c, cDev, bytesC);
  } else {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: skipped automatic copy-back for f64 matmul write slot %d "
          "because target=device; use !$tileoff update host(...) to copy back\n",
          cDev.slot);
    }
  }

  TileOffloadReleaseDeviceArg(aDev);
  TileOffloadReleaseDeviceArg(bDev);
  TileOffloadReleaseDeviceArg(cDev);
}

static CUdeviceptr TileOffloadReserveReductionBuffer(
    TileOffloadDeviceAllocation &allocation, std::size_t requiredBytes,
    TileOffloadReductionBufferStats &stats, const char *bufferName) {
  if (requiredBytes == 0) {
    std::fprintf(stderr,
        "TileOffload error: requested zero bytes for reduction %s buffer\n",
        bufferName);
    std::abort();
  }

  if (allocation.ptr && allocation.bytes >= requiredBytes) {
    ++stats.reuses;
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: reusing reduction %s buffer device=0x%llx "
          "capacity=%zu required=%zu\n",
          bufferName, static_cast<unsigned long long>(allocation.ptr),
          allocation.bytes, requiredBytes);
    }
    return allocation.ptr;
  }

  std::size_t oldBytes = allocation.bytes;
  if (allocation.ptr) {
    TileOffloadSynchronizeActiveContext();
    TILEOFF_CUDA_CHECK(cuMemFree(allocation.ptr));
  }

  allocation = {};
  TILEOFF_CUDA_CHECK(cuMemAlloc(&allocation.ptr, requiredBytes));
  allocation.bytes = requiredBytes;
  ++stats.allocations;
  if (oldBytes != 0)
    ++stats.growths;

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: %s reduction %s buffer device=0x%llx "
        "capacity=%zu\n",
        oldBytes == 0 ? "allocated" : "grew", bufferName,
        static_cast<unsigned long long>(allocation.ptr), allocation.bytes);
  }

  return allocation.ptr;
}

template <typename Real>
static Real TileOffloadReductionIdentity(TileOffloadKernelDesc::ReductionOperator op) {
  switch (op) {
  case TileOffloadKernelDesc::ReductionOperator::Add:
    return Real{0};
  case TileOffloadKernelDesc::ReductionOperator::Multiply:
    return Real{1};
  case TileOffloadKernelDesc::ReductionOperator::Min:
    if constexpr (std::numeric_limits<Real>::has_infinity)
      return std::numeric_limits<Real>::infinity();
    return std::numeric_limits<Real>::max();
  case TileOffloadKernelDesc::ReductionOperator::Max:
    if constexpr (std::numeric_limits<Real>::has_infinity)
      return -std::numeric_limits<Real>::infinity();
    return std::numeric_limits<Real>::lowest();
  }
  std::abort();
}

template <typename Real>
static Real TileOffloadApplyReduction(
    TileOffloadKernelDesc::ReductionOperator op, Real lhs, Real rhs) {
  switch (op) {
  case TileOffloadKernelDesc::ReductionOperator::Add:
    if constexpr (std::is_integral_v<Real>) {
      using Unsigned = std::make_unsigned_t<Real>;
      return static_cast<Real>(
          static_cast<Unsigned>(lhs) + static_cast<Unsigned>(rhs));
    }
    return lhs + rhs;
  case TileOffloadKernelDesc::ReductionOperator::Multiply:
    if constexpr (std::is_integral_v<Real>) {
      using Unsigned = std::make_unsigned_t<Real>;
      return static_cast<Real>(
          static_cast<Unsigned>(lhs) * static_cast<Unsigned>(rhs));
    }
    return lhs * rhs;
  case TileOffloadKernelDesc::ReductionOperator::Min:
    if constexpr (std::is_floating_point_v<Real>) {
      // Match arith.minimumf/maximumf and Tile minf/maxf propagate_nan.
      if (std::isnan(lhs)) return lhs;
      if (std::isnan(rhs)) return rhs;
      if (lhs == Real(0) && rhs == Real(0)) return std::signbit(lhs) ? lhs : rhs;
    }
    return rhs < lhs ? rhs : lhs;
  case TileOffloadKernelDesc::ReductionOperator::Max:
    if constexpr (std::is_floating_point_v<Real>) {
      // Match arith.minimumf/maximumf and Tile minf/maxf propagate_nan.
      if (std::isnan(lhs)) return lhs;
      if (std::isnan(rhs)) return rhs;
      if (lhs == Real(0) && rhs == Real(0)) return std::signbit(lhs) ? rhs : lhs;
    }
    return rhs > lhs ? rhs : lhs;
  }
  std::abort();
}

template <typename Real>
static bool TileOffloadEnqueueReductionOnDevice(const TileOffloadKernelDesc *primaryDesc,
    TileOffloadReductionWorkspace &workspace, CUdeviceptr dPartials,
    unsigned partialCount, CUdeviceptr *result) {
  if (!primaryDesc || primaryDesc->reductionStageId < 0)
    return false;

  int32_t stageKernelId = primaryDesc->reductionStageId;
  const TileOffloadKernelDesc *stageDesc = TileOffloadLookupKernelDesc(stageKernelId);

  if (!stageDesc) {
    std::fprintf(stderr,
        "TileOffload error: reduction kernel id %d references missing stage "
        "kernel id %d\n",
        primaryDesc->id, stageKernelId);
    std::abort();
  }

  if (stageDesc->kind != "reduction_stage1d") {
    std::fprintf(stderr,
        "TileOffload error: reduction stage kernel id %d has unexpected kind "
        "'%s'\n",
        stageKernelId, stageDesc->kind.c_str());
    std::abort();
  }

  if (stageDesc->reductionOp != primaryDesc->reductionOp) {
    std::fprintf(stderr,
        "TileOffload error: primary reduction kernel id %d and stage kernel id %d "
        "use different reduction operators\n",
        primaryDesc->id, stageKernelId);
    std::abort();
  }

  int32_t stageBlock = stageDesc->tileX;
  if (stageBlock <= 1) {
    std::fprintf(stderr,
        "TileOffload error: reduction stage kernel id %d requires tile_x > 1, "
        "got %d\n",
        stageKernelId, stageBlock);
    std::abort();
  }

  CUfunction stageFn = getKernelFunction(stageKernelId);
  unsigned stageCudaBlockX = TileOffloadCudaThreadsPerCTA(stageKernelId);

  TileOffloadValidateSupportedHiddenPtrArgCount(stageKernelId);
  if (stageDesc->backend != "cuda-tile")
    TileOffloadValidateCudaBlockSize(stageFn, stageKernelId, stageCudaBlockX);

  CUdeviceptr current = dPartials;
  CUdeviceptr next = 0;

  // Validate before the first narrowing conversion (including scratch sizing).
  if (partialCount == 0 ||
      partialCount >
          static_cast<unsigned>(std::numeric_limits<int32_t>::max())) {
    std::fprintf(
        stderr, "TileOffload error: reduction stage extent exceeds i32 or is zero\n");
    std::abort();
  }
  if (partialCount > 1) {
    unsigned scratchElements = TileOffloadCdiv(
        static_cast<int32_t>(partialCount), stageBlock, "scratch Elements");
    std::size_t scratchBytes =
        TileOffloadCheckedMul(static_cast<std::size_t>(scratchElements), sizeof(Real),
            "hierarchical reduction scratch buffer");
    next = TileOffloadReserveReductionBuffer(
        workspace.scratch, scratchBytes, workspace.scratchStats, "scratch");
  }

  while (partialCount > 1) {
    if (partialCount >
        static_cast<unsigned>(std::numeric_limits<int32_t>::max())) {
      std::fprintf(stderr, "TileOffload error: reduction stage extent exceeds i32\n");
      std::abort();
    }
    int32_t stageExtent = static_cast<int32_t>(partialCount);
    unsigned outputCount = TileOffloadCdiv(stageExtent, stageBlock, "output Count");
    TileOffloadHiddenTritonArgs hidden;

    void *tritonArgs[] = {
        &current,
        &next,
        &stageExtent,
        &hidden.hidden0,
        &hidden.hidden1,
    };
    void *tileArgs[] = {&current, &next, &stageExtent};
    void **args = stageDesc->tritonHiddenPtrArgs == 0 ? tileArgs : tritonArgs;

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: launch reduction stage kernel id=%d input_count=%u "
          "output_count=%u tile=%d cuda_block=%u\n",
          stageKernelId, partialCount, outputCount, stageBlock,
          stageCudaBlockX);
    }

    unsigned dynamicSharedBytes =
        TileOffloadReductionDynamicSharedBytes(stageDesc, stageBlock);

    TileOffloadConfigureDynamicSharedMemory(
        stageFn, stageKernelId, dynamicSharedBytes);

    TILEOFF_CUDA_CHECK(cuLaunchKernel(stageFn, outputCount, 1, 1, stageCudaBlockX,
        1, 1, dynamicSharedBytes, TileOffloadActiveContextState().stream, args,
        nullptr));
    ++workspace.stageLaunches;

    CUdeviceptr oldCurrent = current;
    current = next;
    next = oldCurrent;
    partialCount = outputCount;
  }

  *result = current;

  return true;
}

template <typename Real>
static bool TileOffloadFinalizeReductionOnDevice(const TileOffloadKernelDesc *desc,
    TileOffloadReductionWorkspace &workspace, CUdeviceptr partials, unsigned count,
    Real *result) {
  CUdeviceptr deviceResult = 0;
  if (!TileOffloadEnqueueReductionOnDevice<Real>(
          desc, workspace, partials, count, &deviceResult))
    return false;
  TileOffloadWaitForRuntimeStream();
  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(result, deviceResult, sizeof(Real)));
  return true;
}

extern "C" void __tileoff_launch_reduce_f32_v2(int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, float *read0, float *read1, float *result,
    float initialValue, int32_t extentX) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(
        stderr, "TileOffload error: null pointer in __tileoff_launch_reduce_f32_v2\n");
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "TileOffload error: reduction f32 supports one or two read arrays, got %d\n",
        numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  TileOffloadReductionWorkspace &workspace = TileOffloadGetReductionWorkspace();

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(float);

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, 1, read1)
      : TILEOFF_PACK_TARGET_HOST;

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, bytes, read0Target, 0);

  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;

  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(float);
  CUdeviceptr dPartials = TileOffloadReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[8];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes = TileOffloadReductionDynamicSharedBytes(desc, blockX);

  TileOffloadConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  float reducedValue = TileOffloadReductionIdentity<float>(desc->reductionOp);

  if (!TileOffloadFinalizeReductionOnDevice<float>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    TileOffloadWaitForRuntimeStream();

    std::vector<float> partials(gridX);
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(partials.data(), dPartials, partialBytes));

    for (float value : partials)
      reducedValue =
          TileOffloadApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = TileOffloadApplyReduction(desc->reductionOp, initialValue, reducedValue);

  TileOffloadReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);
}

extern "C" void __tileoff_launch_reduce_f64_v2(int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, double *read0, double *read1, double *result,
    double initialValue, int32_t extentX) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(
        stderr, "TileOffload error: null pointer in __tileoff_launch_reduce_f64_v2\n");
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "TileOffload error: reduction f64 supports one or two read arrays, got %d\n",
        numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  TileOffloadReductionWorkspace &workspace = TileOffloadGetReductionWorkspace();

  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);

  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(double);

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, 1, read1)
      : TILEOFF_PACK_TARGET_HOST;

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, bytes, read0Target, 0);

  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;

  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(double);
  CUdeviceptr dPartials = TileOffloadReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;

  void *args[8];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes =
      TileOffloadReductionF64DynamicSharedBytes(desc, blockX);

  TileOffloadConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  double reducedValue = TileOffloadReductionIdentity<double>(desc->reductionOp);

  if (!TileOffloadFinalizeReductionOnDevice<double>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    TileOffloadWaitForRuntimeStream();

    std::vector<double> partials(gridX);
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(partials.data(), dPartials, partialBytes));

    for (double value : partials)
      reducedValue =
          TileOffloadApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = TileOffloadApplyReduction(desc->reductionOp, initialValue, reducedValue);

  TileOffloadReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);
}

template <typename Integer>
static void TileOffloadLaunchReduceIntegerV2(const char *abiName,
    const char *typeName, int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, Integer *read0, Integer *read1, Integer *result,
    Integer initialValue, int32_t extentX) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(stderr, "TileOffload error: null pointer in %s\n", abiName);
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "TileOffload error: reduction %s supports one or two read arrays, got %d\n",
        typeName, numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const TileOffloadKernelDesc *desc = TileOffloadLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "TileOffload error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  TileOffloadReductionWorkspace &workspace = TileOffloadGetReductionWorkspace();
  unsigned gridX = TileOffloadCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = TileOffloadCudaThreadsPerCTA(kernelId);
  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(Integer);

  int32_t read0Target = TileOffloadEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? TileOffloadEffectivePackTargetForSlot(desc, 1, read1)
      : TILEOFF_PACK_TARGET_HOST;

  TileOffloadDeviceArg read0Dev =
      TileOffloadPrepareReadBuffer(read0, bytes, read0Target, 0);
  TileOffloadDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = TileOffloadPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(Integer);
  CUdeviceptr dPartials = TileOffloadReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  TileOffloadValidateSupportedHiddenPtrArgCount(kernelId);
  TileOffloadHiddenTritonArgs hidden;
  void *args[8];
  int argCount = 0;
  args[argCount++] = &dRead0;
  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;
  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  TileOffloadValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes =
      TileOffloadReductionIntegerDynamicSharedBytes(desc, blockX, sizeof(Integer));

  TileOffloadConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  TILEOFF_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, TileOffloadActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  Integer reducedValue = TileOffloadReductionIdentity<Integer>(desc->reductionOp);
  if (!TileOffloadFinalizeReductionOnDevice<Integer>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    TileOffloadWaitForRuntimeStream();

    std::vector<Integer> partials(gridX);
    TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(partials.data(), dPartials, partialBytes));
    for (Integer value : partials)
      reducedValue =
          TileOffloadApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = TileOffloadApplyReduction(desc->reductionOp, initialValue, reducedValue);

  TileOffloadReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    TileOffloadReleaseDeviceArg(read1Dev);
}

#define TILEOFF_DEFINE_INTEGER_REDUCTION(BITS, TYPE) \
  extern "C" void __tileoff_launch_reduce_i##BITS##_v2(int32_t kernelId, \
      int32_t blockX, int32_t numReadArrays, TYPE *read0, TYPE *read1, \
      TYPE *result, TYPE initialValue, int32_t extentX) { \
    TileOffloadLaunchReduceIntegerV2<TYPE>("__tileoff_launch_reduce_i" #BITS "_v2", \
        "i" #BITS, kernelId, blockX, numReadArrays, read0, read1, result, \
        initialValue, extentX); \
  }

TILEOFF_DEFINE_INTEGER_REDUCTION(8, int8_t)
TILEOFF_DEFINE_INTEGER_REDUCTION(16, int16_t)
TILEOFF_DEFINE_INTEGER_REDUCTION(32, int32_t)
TILEOFF_DEFINE_INTEGER_REDUCTION(64, int64_t)

#undef TILEOFF_DEFINE_INTEGER_REDUCTION

extern "C" void __tileoff_get_reduction_workspace_stats_v1(
    uint64_t *primaryLaunches, uint64_t *stageLaunches,
    uint64_t *partialAllocations, uint64_t *partialGrowths,
    uint64_t *partialReuses, uint64_t *partialCapacityBytes,
    uint64_t *scratchAllocations, uint64_t *scratchGrowths,
    uint64_t *scratchReuses, uint64_t *scratchCapacityBytes) {
  TILEOFF_REGISTRY_GUARD();
  TileOffloadReductionWorkspace workspace = TileOffloadAggregateReductionWorkspaceStats();
  if (primaryLaunches)
    *primaryLaunches = workspace.primaryLaunches;
  if (stageLaunches)
    *stageLaunches = workspace.stageLaunches;
  if (partialAllocations)
    *partialAllocations = workspace.partialStats.allocations;
  if (partialGrowths)
    *partialGrowths = workspace.partialStats.growths;
  if (partialReuses)
    *partialReuses = workspace.partialStats.reuses;
  if (partialCapacityBytes)
    *partialCapacityBytes = static_cast<uint64_t>(workspace.partials.bytes);
  if (scratchAllocations)
    *scratchAllocations = workspace.scratchStats.allocations;
  if (scratchGrowths)
    *scratchGrowths = workspace.scratchStats.growths;
  if (scratchReuses)
    *scratchReuses = workspace.scratchStats.reuses;
  if (scratchCapacityBytes)
    *scratchCapacityBytes = static_cast<uint64_t>(workspace.scratch.bytes);
}

// Memory management functions to help with cached data and data lifetimes
extern "C" void __tileoff_update_host(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_host ignored null pointer\n");
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "TileOffload error: update_host has no cached allocation for %p; "
        "use create/copyin/update_device first\n",
        hostPtr);
    std::abort();
  }

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyDtoH(hostPtr, it->second.ptr, it->second.bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: update_host host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        it->second.bytes);
  }
}

extern "C" void __tileoff_update_device(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: update_device ignored null pointer\n");
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "TileOffload error: update_device has no cached allocation for %p; "
        "use a sized update/create directive first\n",
        hostPtr);
    std::abort();
  }

  TILEOFF_CUDA_CHECK(TileOffloadMemcpyHtoD(it->second.ptr, hostPtr, it->second.bytes));

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: update_device host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(it->second.ptr), it->second.bytes);
  }
}

extern "C" void __tileoff_release(void *hostPtr) {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  if (!hostPtr) {
    if (TileOffloadDebugEnabled())
      std::fprintf(stderr, "TileOffload: release ignored null pointer\n");
    return;
  }

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: release ignored; no cached allocation for %p\n", hostPtr);
    }
    return;
  }

  if (it->second.dataRegionReferences != 0) {
    std::fprintf(stderr,
        "TileOffload error: release cannot release host=%p while it is owned by "
        "%zu data region(s); exit the owning region first\n",
        hostPtr, it->second.dataRegionReferences);
    std::abort();
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  TileOffloadSynchronizeActiveContext();

  TILEOFF_CUDA_CHECK(cuMemFree(devicePtr));

  cache.erase(it);

  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr, "TileOffload: release host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __tileoff_release_all() {
  TILEOFF_RUNTIME_GUARD();
  TileOffloadCurrentContextGuard contextGuard;
  TileOffloadEnsureCurrentContext();

  auto &cache = TileOffloadActiveContextState().deviceCache;
  auto &regions = TileOffloadActiveContextState().dataRegions;
  if (!regions.empty()) {
    std::fprintf(stderr,
        "TileOffload error: release_all cannot be used while %zu data region(s) "
        "are active; exit the regions first\n",
        regions.size());
    std::abort();
  }
  if (TileOffloadDebugEnabled()) {
    std::fprintf(stderr,
        "TileOffload: release_all releasing %zu cached allocations\n", cache.size());
  }

  if (!cache.empty())
    TileOffloadSynchronizeActiveContext();
  for (auto &entry : cache) {
    void *hostPtr = entry.first;
    TileOffloadDeviceAllocation &allocation = entry.second;

    if (TileOffloadDebugEnabled()) {
      std::fprintf(stderr,
          "TileOffload: release_all host=%p device=0x%llx bytes=%zu\n", hostPtr,
          static_cast<unsigned long long>(allocation.ptr), allocation.bytes);
    }
    if (allocation.ptr)
      TILEOFF_CUDA_CHECK(cuMemFree(allocation.ptr));
  }

  cache.clear();
}

static void TileOffloadRegisterEmbeddedDeviceBundle(const void *const *imageData,
    const std::size_t *imageSizes, const int32_t *imageKinds,
    std::size_t imageCount, const char *jsonData, std::size_t jsonSize) {
  if (!imageData || !imageSizes || !imageKinds || imageCount == 0 ||
      !jsonData || jsonSize == 0) {
    std::fprintf(stderr, "TileOffload error: invalid embedded kernel bundle\n");
    std::abort();
  }
  for (std::size_t i = 0; i < imageCount; ++i) {
    if (!imageData[i] || imageSizes[i] == 0 ||
        (imageKinds[i] != TileOffloadEmbeddedKernelBundle::PTX &&
            imageKinds[i] != TileOffloadEmbeddedKernelBundle::Cubin &&
            imageKinds[i] != TileOffloadEmbeddedKernelBundle::HSACO)) {
      std::fprintf(
          stderr, "TileOffload error: invalid embedded device image entry %zu\n", i);
      std::abort();
    }
  }

  TileOffloadEmbeddedKernelBundle bundle;
  bundle.imageData.assign(imageData, imageData + imageCount);
  bundle.imageSize.assign(imageSizes, imageSizes + imageCount);
  bundle.imageKind.assign(imageKinds, imageKinds + imageCount);
  bundle.jsonData = jsonData;
  bundle.jsonSize = jsonSize;
  TileOffloadGetEmbeddedKernelBundles().push_back(std::move(bundle));
}

extern "C" void __tileoff_register_embedded_device_bundle(
    const void *const *imageData, const std::size_t *imageSizes,
    const int32_t *imageKinds, std::size_t imageCount, const char *jsonData,
    std::size_t jsonSize) {
  TILEOFF_REGISTRY_GUARD();
  TileOffloadRegisterEmbeddedDeviceBundle(
      imageData, imageSizes, imageKinds, imageCount, jsonData, jsonSize);
}

extern "C" void __tileoff_register_embedded_kernel_bundle(
    const char *const *ptxData, std::size_t const *ptxSizes,
    std::size_t ptxCount, const char *jsonData, std::size_t jsonSize) {
  TILEOFF_REGISTRY_GUARD();
  std::vector<const void *> imageData(ptxCount);
  for (std::size_t i = 0; i < ptxCount; ++i)
    imageData[i] = ptxData ? ptxData[i] : nullptr;
  std::vector<int32_t> imageKinds(ptxCount, TileOffloadEmbeddedKernelBundle::PTX);
  TileOffloadRegisterEmbeddedDeviceBundle(imageData.data(), ptxSizes,
      imageKinds.data(), ptxCount, jsonData, jsonSize);
}

extern "C" void __tileoff_register_embedded_kernels(const char *ptxData,
    std::size_t ptxSize, const char *jsonData, std::size_t jsonSize) {
  TILEOFF_REGISTRY_GUARD();
  const void *imageData[] = {ptxData};
  std::size_t imageSizes[] = {ptxSize};
  int32_t imageKinds[] = {TileOffloadEmbeddedKernelBundle::PTX};
  TileOffloadRegisterEmbeddedDeviceBundle(
      imageData, imageSizes, imageKinds, 1, jsonData, jsonSize);
}

// For profiling
extern "C" void tileoff_profile_compute_begin() {
  TILEOFF_PROFILE_PUSH("TileOffload.compute");
}

extern "C" void tileoff_profile_compute_end() { TILEOFF_PROFILE_POP(); }

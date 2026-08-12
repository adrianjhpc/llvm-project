#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static void fnaccCudaCheck(
    CUresult result, const char *expr, const char *file, int line) {
  if (result == CUDA_SUCCESS)
    return;

  const char *name = nullptr;
  const char *desc = nullptr;

  cuGetErrorName(result, &name);
  cuGetErrorString(result, &desc);

  std::fprintf(stderr,
      "FNACC CUDA driver error at %s:%d while executing %s: %s: %s\n", file,
      line, expr, name ? name : "<unknown>", desc ? desc : "<no description>");

  std::abort();
}

#define FNACC_CUDA_CHECK(expr) \
  do { \
    fnaccCudaCheck((expr), #expr, __FILE__, __LINE__); \
  } while (false)

static constexpr const char *FNACC_RUNTIME_BUILD_ID =
    "FNACC_RUNTIME_BUILD_ID_matmul_cdiv_grid_v2";

static std::size_t fnaccCheckedMul(
    std::size_t a, std::size_t b, const char *what) {
  if (a != 0 && b > static_cast<std::size_t>(-1) / a) {
    std::fprintf(
        stderr, "FNACC error: size overflow while computing %s\n", what);
    std::abort();
  }

  return a * b;
}

static void fnaccConfigureDynamicSharedMemory(
    CUfunction fn, int32_t kernelId, unsigned dynamicSharedBytes) {
  if (dynamicSharedBytes == 0)
    return;

  // 48 KiB is usually available without opt-in on many NVIDIA GPUs.
  // Above that, opt in if the device/function supports it.
  if (dynamicSharedBytes > 49152) {
    CUresult result =
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(dynamicSharedBytes));

    if (result != CUDA_SUCCESS) {
      const char *name = nullptr;
      const char *desc = nullptr;
      cuGetErrorName(result, &name);
      cuGetErrorString(result, &desc);

      std::fprintf(stderr,
          "FNACC error: could not set dynamic shared memory size for kernel "
          "id %d to %u bytes: %s: %s\n",
          kernelId, dynamicSharedBytes, name ? name : "<unknown>",
          desc ? desc : "<no description>");
      std::abort();
    }
  }
}

static unsigned fnaccGetEnvUnsignedAllowZero(
    const char *name, unsigned fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;

  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);

  if (end == value || *end != '\0' ||
      parsed >
          static_cast<unsigned long>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr, "FNACC error: invalid %s value '%s'\n", name, value);
    std::abort();
  }

  return static_cast<unsigned>(parsed);
}

static std::size_t fnaccCheckedBytes2D(
    int32_t dim0, int32_t dim1, std::size_t elemBytes, const char *what) {
  if (dim0 < 0 || dim1 < 0) {
    std::fprintf(stderr,
        "FNACC error: negative dimension while computing %s: (%d,%d)\n", what,
        dim0, dim1);
    std::abort();
  }

  std::size_t elements = fnaccCheckedMul(
      static_cast<std::size_t>(dim0), static_cast<std::size_t>(dim1), what);

  return fnaccCheckedMul(elements, elemBytes, what);
}

static void fnaccValidateCudaBlockSize(
    CUfunction fn, int32_t kernelId, unsigned cudaBlockX) {
  int maxThreadsPerBlock = 0;
  FNACC_CUDA_CHECK(cuFuncGetAttribute(
      &maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));

  if (cudaBlockX > static_cast<unsigned>(maxThreadsPerBlock)) {
    std::fprintf(stderr,
        "FNACC error: kernel id %d requested CUDA block size %u, "
        "but function max_threads_per_block is %d\n",
        kernelId, cudaBlockX, maxThreadsPerBlock);
    std::abort();
  }
}

static int fnaccGetCudaDeviceOrdinal() {
  const char *value = std::getenv("FNACC_CUDA_DEVICE");
  if (!value || value[0] == '\0')
    return 0;

  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0) {
    std::fprintf(
        stderr, "FNACC error: invalid FNACC_CUDA_DEVICE value '%s'\n", value);
    std::abort();
  }

  return static_cast<int>(parsed);
}

// -------------------------------------------------------------------------- //
// Tiny dependency-free JSON helpers
// -------------------------------------------------------------------------- //
//
// These intentionally parse only the JSON shape emitted by the FNACC compiler.
// This is not a general-purpose JSON parser.
//
// Expected generated object form:
//
// {
//   "id": 5,
//   "name": "fnacc_kernel_5",
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

  char *end = nullptr;
  long value = std::strtol(cursor, &end, 10);
  if (end == cursor)
    return false;

  out = static_cast<int32_t>(value);
  return true;
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

static void fnaccValidateContiguousDescriptor(const char *operationName,
    int64_t elementBytes, int32_t rank, int64_t extent0, int64_t extent1,
    int64_t extent2, int64_t stride0, int64_t stride1, int64_t stride2) {
  if (rank < 1 || rank > 3) {
    std::fprintf(stderr, "FNACC error: %s received unsupported rank %d\n",
        operationName, rank);
    std::abort();
  }

  if (elementBytes <= 0) {
    std::fprintf(stderr, "FNACC error: %s received invalid element size %lld\n",
        operationName, static_cast<long long>(elementBytes));
    std::abort();
  }

  int64_t expected0 = elementBytes;
  int64_t expected1 = elementBytes * extent0;
  int64_t expected2 = elementBytes * extent0 * extent1;

  bool contiguous = true;

  if (rank >= 1 && stride0 != expected0)
    contiguous = false;
  if (rank >= 2 && stride1 != expected1)
    contiguous = false;
  if (rank >= 3 && stride2 != expected2)
    contiguous = false;

  if (!contiguous) {
    std::fprintf(stderr,
        "FNACC error: %s only supports contiguous assumed-shape arrays; "
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

static constexpr int32_t FNACC_SUPPORTED_SCHEMA_VERSION = 1;
static constexpr int32_t FNACC_PACK_TARGET_HOST = 0;
static constexpr int32_t FNACC_PACK_TARGET_DEVICE = 1;

struct FNACCPackEntry {
  int32_t kernelArgSlot = -1;
  int32_t target = FNACC_PACK_TARGET_HOST;
};

static std::vector<FNACCPackEntry> jsonParsePackEntries(
    const std::string &kernelObjectText) {
  std::vector<FNACCPackEntry> entries;

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

    FNACCPackEntry entry;

    if (!jsonFindInt(objectText, "kernel_arg_slot", entry.kernelArgSlot)) {
      pos = objectEnd;
      continue;
    }

    if (!jsonFindInt(objectText, "target", entry.target))
      entry.target = FNACC_PACK_TARGET_HOST;

    if (entry.target != FNACC_PACK_TARGET_HOST &&
        entry.target != FNACC_PACK_TARGET_DEVICE) {
      std::fprintf(stderr,
          "FNACC warning: invalid pack target %d for slot %d; "
          "defaulting to host\n",
          entry.target, entry.kernelArgSlot);
      entry.target = FNACC_PACK_TARGET_HOST;
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

struct FNACCHiddenTritonArgs {
  // Triton/NVVM-generated PTX currently appends two hidden pointer parameters
  // after the explicit kernel parameters. For the kernels FNACC currently
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

struct FNACCKernelDesc {
  int32_t id = -1;
  std::string name;
  std::string kind = "binary";

  int32_t rank = 1;

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

  // PACK metadata from JSON.
  std::vector<FNACCPackEntry> pack;
};

struct FNACCDeviceAllocation {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
};

struct FNACCKernelRegistry {
  bool initialized = false;

  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;

  std::unordered_map<int32_t, FNACCKernelDesc> kernels;
  std::unordered_map<int32_t, CUfunction> functionCache;

  // Device cache keyed by host pointer.
  std::unordered_map<void *, FNACCDeviceAllocation> deviceCache;
};

static const char *fnaccEmbeddedPtxData = nullptr;
static std::size_t fnaccEmbeddedPtxSize = 0;

static const char *fnaccEmbeddedJsonData = nullptr;
static std::size_t fnaccEmbeddedJsonSize = 0;

static FNACCKernelRegistry fnaccRegistry;

struct FNACCDeviceArg {
  CUdeviceptr ptr = 0;
  bool cached = false;
  int32_t target = FNACC_PACK_TARGET_HOST;
  int32_t slot = -1;
};

static bool fnaccDebugEnabled() {
  const char *value = std::getenv("FNACC_DEBUG");
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool fnaccHasEmbeddedPtx() {
  return fnaccEmbeddedPtxData && fnaccEmbeddedPtxSize > 0;
}

static bool fnaccHasEmbeddedJson() {
  return fnaccEmbeddedJsonData && fnaccEmbeddedJsonSize > 0;
}

static std::string fnaccReadTextFile(const char *path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    std::fprintf(stderr, "FNACC error: could not open file '%s'\n", path);
    std::abort();
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

static std::string fnaccGetPtxText() {
  const char *ptxPath = std::getenv("FNACC_PTX");

  if (ptxPath && ptxPath[0] != '\0') {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: loading PTX from '%s'\n", ptxPath);

    return fnaccReadTextFile(ptxPath);
  }

  if (fnaccHasEmbeddedPtx()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: loading embedded PTX, bytes=%zu\n",
          fnaccEmbeddedPtxSize);
    }

    return std::string(fnaccEmbeddedPtxData, fnaccEmbeddedPtxSize);
  }

  const char *fallback = "fnacc_kernels.ptx";

  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: loading PTX from '%s'\n", fallback);

  return fnaccReadTextFile(fallback);
}

static std::string fnaccGetJsonText() {
  const char *jsonPath = std::getenv("FNACC_KERNELS_JSON");

  if (jsonPath && jsonPath[0] != '\0') {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: loading JSON from '%s'\n", jsonPath);

    return fnaccReadTextFile(jsonPath);
  }

  if (fnaccHasEmbeddedJson()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: loading embedded JSON, bytes=%zu\n",
          fnaccEmbeddedJsonSize);
    }

    return std::string(fnaccEmbeddedJsonData, fnaccEmbeddedJsonSize);
  }

  const char *fallback = "fnacc_kernels.json";

  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: loading JSON from '%s'\n", fallback);

  return fnaccReadTextFile(fallback);
}

static std::unordered_map<int32_t, FNACCKernelDesc>
fnaccParseKernelDescsFromJson(const std::string &json) {
  std::unordered_map<int32_t, FNACCKernelDesc> result;

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
      std::fprintf(stderr, "FNACC error: malformed kernel JSON object\n");
      std::abort();
    }

    std::string objectText = json.substr(objectStart, objectEnd - objectStart);

    FNACCKernelDesc desc;

    if (!jsonFindInt(objectText, "id", desc.id)) {
      pos = objectEnd;
      continue;
    }

    if (!jsonFindString(objectText, "name", desc.name))
      desc.name = "fnacc_kernel_" + std::to_string(desc.id);

    jsonFindString(objectText, "kind", desc.kind);

    jsonFindInt(objectText, "rank", desc.rank);

    jsonFindIntArray3(objectText, "tile", desc.tileX, desc.tileY, desc.tileZ);

    jsonFindInt(objectText, "num_warps", desc.numWarps);
    jsonFindInt(objectText, "threads_per_warp", desc.threadsPerWarp);
    jsonFindInt(objectText, "num_ctas", desc.numCTAs);
    jsonFindInt(objectText, "num_stages", desc.numStages);

    if (!jsonFindInt(
            objectText, "cuda_threads_per_cta", desc.cudaThreadsPerCTA)) {
      desc.cudaThreadsPerCTA = desc.numWarps * desc.threadsPerWarp;
    }

    jsonFindInt(objectText, "triton_hidden_ptr_args", desc.tritonHiddenPtrArgs);

    desc.pack = jsonParsePackEntries(objectText);

    if (desc.numWarps <= 0)
      desc.numWarps = 1;

    if (desc.threadsPerWarp <= 0)
      desc.threadsPerWarp = 32;

    if (desc.cudaThreadsPerCTA <= 0)
      desc.cudaThreadsPerCTA = desc.numWarps * desc.threadsPerWarp;

    if (desc.cudaThreadsPerCTA <= 0)
      desc.cudaThreadsPerCTA = 32;

    if (desc.tritonHiddenPtrArgs < 0)
      desc.tritonHiddenPtrArgs = 0;

    result[desc.id] = desc;

    pos = objectEnd;
  }

  return result;
}

static void fnaccCleanup() {
  if (!fnaccRegistry.initialized)
    return;

  if (fnaccRegistry.context)
    cuCtxSetCurrent(fnaccRegistry.context);

  for (auto &entry : fnaccRegistry.deviceCache) {
    if (entry.second.ptr)
      cuMemFree(entry.second.ptr);
  }

  fnaccRegistry.deviceCache.clear();
  fnaccRegistry.functionCache.clear();
  fnaccRegistry.kernels.clear();

  if (fnaccRegistry.module) {
    cuModuleUnload(fnaccRegistry.module);
    fnaccRegistry.module = nullptr;
  }

  if (fnaccRegistry.context) {
    cuDevicePrimaryCtxRelease(fnaccRegistry.device);
    fnaccRegistry.context = nullptr;
  }

  fnaccRegistry.initialized = false;
}

static unsigned fnaccMatmulDynamicSharedBytes(const FNACCKernelDesc *desc,
    int32_t blockX, int32_t blockY, int32_t blockK) {
  int32_t stages = 1;
  if (desc && desc->numStages > 0)
    stages = desc->numStages;

  std::size_t aElems = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockK),
      "matmul dynamic shared A tile elements");

  std::size_t bElems = fnaccCheckedMul(static_cast<std::size_t>(blockK),
      static_cast<std::size_t>(blockY),
      "matmul dynamic shared B tile elements");

  std::size_t elems =
      fnaccCheckedMul(aElems + bElems, static_cast<std::size_t>(stages),
          "matmul dynamic shared staged tile elements");

  std::size_t bytes =
      fnaccCheckedMul(elems, sizeof(float), "matmul dynamic shared bytes");

  // Align to 256 bytes.
  bytes = (bytes + 255) & ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple A/B tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: matmul dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  // Optional override for experiments, but do not allow values below the
  // computed requirement. A too-small dynamic shared memory size can cause
  // illegal GPU memory accesses.
  if (const char *value = std::getenv("FNACC_MATMUL_SHARED_BYTES")) {
    if (value[0] != '\0') {
      unsigned requested = fnaccGetEnvUnsignedAllowZero(
          "FNACC_MATMUL_SHARED_BYTES", requiredBytes);

      if (requested < requiredBytes) {
        std::fprintf(stderr,
            "FNACC error: FNACC_MATMUL_SHARED_BYTES=%u is smaller than the "
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

static void fnaccEnsureInitialized() {
  if (fnaccRegistry.initialized)
    return;

  if (fnaccDebugEnabled()) {
    std::fprintf(
        stderr, "FNACC: runtime build id: %s\n", FNACC_RUNTIME_BUILD_ID);
  }

  std::string ptx = fnaccGetPtxText();

  FNACC_CUDA_CHECK(cuInit(0));

  int ordinal = fnaccGetCudaDeviceOrdinal();

  FNACC_CUDA_CHECK(cuDeviceGet(&fnaccRegistry.device, ordinal));

  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: using CUDA device ordinal %d\n", ordinal);

  FNACC_CUDA_CHECK(
      cuDevicePrimaryCtxRetain(&fnaccRegistry.context, fnaccRegistry.device));

  FNACC_CUDA_CHECK(cuCtxSetCurrent(fnaccRegistry.context));

  FNACC_CUDA_CHECK(cuModuleLoadDataEx(
      &fnaccRegistry.module, ptx.c_str(), 0, nullptr, nullptr));

  std::string json = fnaccGetJsonText();

  int32_t schemaVersion = 0;
  if (!jsonFindInt(json, "fnacc_schema_version", schemaVersion)) {
    std::fprintf(
        stderr, "FNACC error: kernel JSON is missing fnacc_schema_version\n");
    std::abort();
  }

  if (schemaVersion != FNACC_SUPPORTED_SCHEMA_VERSION) {
    std::fprintf(stderr,
        "FNACC error: unsupported kernel JSON schema version %d; "
        "runtime supports version %d\n",
        schemaVersion, FNACC_SUPPORTED_SCHEMA_VERSION);
    std::abort();
  }

  fnaccRegistry.kernels = fnaccParseKernelDescsFromJson(json);

  if (fnaccDebugEnabled()) {
    for (const auto &entry : fnaccRegistry.kernels) {
      const FNACCKernelDesc &desc = entry.second;

      std::fprintf(stderr,
          "FNACC: registered kernel id %d -> '%s' "
          "kind=%s rank=%d tile=(%d,%d,%d) "
          "warps=%d threads_per_warp=%d "
          "cuda_threads_per_cta=%d hidden_ptr_args=%d\n",
          desc.id, desc.name.c_str(), desc.kind.c_str(), desc.rank, desc.tileX,
          desc.tileY, desc.tileZ, desc.numWarps, desc.threadsPerWarp,
          desc.cudaThreadsPerCTA, desc.tritonHiddenPtrArgs);
      for (const FNACCPackEntry &entry : desc.pack) {
        std::fprintf(stderr, "FNACC:   pack slot %d -> %s\n",
            entry.kernelArgSlot,
            entry.target == FNACC_PACK_TARGET_DEVICE ? "device" : "host");
      }
    }
  }

  fnaccRegistry.initialized = true;
  std::atexit(fnaccCleanup);
}

static void fnaccEnsureCurrentContext() {
  fnaccEnsureInitialized();

  if (!fnaccRegistry.context) {
    std::fprintf(stderr, "FNACC error: CUDA context is null\n");
    std::abort();
  }

  FNACC_CUDA_CHECK(cuCtxSetCurrent(fnaccRegistry.context));

  if (fnaccDebugEnabled()) {
    CUcontext current = nullptr;
    FNACC_CUDA_CHECK(cuCtxGetCurrent(&current));

    std::fprintf(stderr,
        "FNACC: current CUDA context = %p, registry context = %p\n",
        static_cast<void *>(current),
        static_cast<void *>(fnaccRegistry.context));
  }
}

static const FNACCKernelDesc *fnaccLookupKernelDesc(int32_t kernelId) {
  fnaccEnsureCurrentContext();

  auto it = fnaccRegistry.kernels.find(kernelId);
  if (it == fnaccRegistry.kernels.end())
    return nullptr;

  return &it->second;
}

static int32_t fnaccTritonHiddenPtrArgCount(int32_t kernelId) {
  if (const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId))
    return desc->tritonHiddenPtrArgs;

  return 2;
}

static void fnaccValidateSupportedHiddenPtrArgCount(int32_t kernelId) {
  int32_t count = fnaccTritonHiddenPtrArgCount(kernelId);

  if (count == 2)
    return;

  std::fprintf(stderr,
      "FNACC error: kernel id %d requires %d Triton hidden pointer "
      "but this runtime currently supports exactly 2. "
      "This usually means the Triton/PTX generation pipeline changed and the "
      "FNACC runtime ABI must be updated.\n",
      kernelId, count);
  std::abort();
}

static std::optional<int32_t> fnaccExplicitPackTargetForSlot(
    const FNACCKernelDesc *desc, int32_t slot) {
  if (!desc)
    return std::nullopt;

  for (const FNACCPackEntry &entry : desc->pack) {
    if (entry.kernelArgSlot == slot)
      return entry.target;
  }

  return std::nullopt;
}

static bool fnaccHostPointerIsPresentOnDevice(void *hostPtr) {
  if (!hostPtr)
    return false;

  return fnaccRegistry.deviceCache.find(hostPtr) !=
      fnaccRegistry.deviceCache.end();
}

static int32_t fnaccEffectivePackTargetForSlot(
    const FNACCKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (auto explicitTarget = fnaccExplicitPackTargetForSlot(desc, slot))
    return *explicitTarget;

  // Present-if-cached default:
  //
  // If the user has already created a persistent device allocation with
  // enter data/update device/pack(...:device), then later launches can omit
  // pack(...:device). We use the cached device allocation automatically.
  if (fnaccHostPointerIsPresentOnDevice(hostPtr))
    return FNACC_PACK_TARGET_DEVICE;

  return FNACC_PACK_TARGET_HOST;
}

static const char *fnaccPackTargetSourceName(
    const FNACCKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (fnaccExplicitPackTargetForSlot(desc, slot))
    return "explicit";

  if (fnaccHostPointerIsPresentOnDevice(hostPtr))
    return "present";

  return "default-host";
}

static const char *fnaccPackTargetName(int32_t target) {
  return target == FNACC_PACK_TARGET_DEVICE ? "device" : "host";
}

static FNACCDeviceArg fnaccMakeTemporaryDeviceBuffer(void *hostPtr,
    std::size_t bytes, bool copyHostToDevice, int32_t slot, const char *role) {
  FNACCDeviceArg arg;
  arg.cached = false;
  arg.target = FNACC_PACK_TARGET_HOST;
  arg.slot = slot;

  FNACC_CUDA_CHECK(cuMemAlloc(&arg.ptr, bytes));

  if (copyHostToDevice)
    FNACC_CUDA_CHECK(cuMemcpyHtoD(arg.ptr, hostPtr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: temporary device buffer for %s slot %d: "
        "host=%p device=0x%llx bytes=%zu copy_in=%s\n",
        role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes,
        copyHostToDevice ? "yes" : "no");
  }

  return arg;
}

static FNACCDeviceArg fnaccGetCachedDeviceBuffer(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnMiss, int32_t slot,
    const char *role) {
  FNACCDeviceArg arg;
  arg.cached = true;
  arg.target = FNACC_PACK_TARGET_DEVICE;
  arg.slot = slot;

  auto it = fnaccRegistry.deviceCache.find(hostPtr);

  bool needAllocate = false;

  if (it == fnaccRegistry.deviceCache.end()) {
    needAllocate = true;
  } else if (it->second.bytes != bytes) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: cache size mismatch for %s slot %d host=%p; "
          "old bytes=%zu new bytes=%zu, reallocating\n",
          role, slot, hostPtr, it->second.bytes, bytes);
    }

    FNACC_CUDA_CHECK(cuMemFree(it->second.ptr));
    fnaccRegistry.deviceCache.erase(it);
    needAllocate = true;
  }

  if (needAllocate) {
    FNACCDeviceAllocation allocation;
    allocation.bytes = bytes;
    FNACC_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnMiss)
      FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));

    auto inserted = fnaccRegistry.deviceCache.emplace(hostPtr, allocation);
    arg.ptr = inserted.first->second.ptr;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: cache miss for %s slot %d target=device: "
          "host=%p device=0x%llx bytes=%zu copy_in=%s\n",
          role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes,
          copyHostToDeviceOnMiss ? "yes" : "no");
    }
  } else {
    arg.ptr = it->second.ptr;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: cache hit for %s slot %d target=device: "
          "host=%p device=0x%llx bytes=%zu\n",
          role, slot, hostPtr, static_cast<unsigned long long>(arg.ptr), bytes);
    }
  }

  return arg;
}

static FNACCDeviceArg fnaccPrepareReadBuffer(
    void *hostPtr, std::size_t bytes, int32_t target, int32_t slot) {
  if (target == FNACC_PACK_TARGET_DEVICE) {
    // Device target means cache/reuse device allocation. Copy in only on miss.
    return fnaccGetCachedDeviceBuffer(static_cast<void *>(hostPtr), bytes,
        /*copyHostToDeviceOnMiss=*/true, slot, "read");
  }

  return fnaccMakeTemporaryDeviceBuffer(static_cast<void *>(hostPtr), bytes,
      /*copyHostToDevice=*/true, slot, "read");
}

static FNACCDeviceArg fnaccPrepareWriteBuffer(
    void *hostPtr, std::size_t bytes, int32_t target, int32_t slot) {
  if (target == FNACC_PACK_TARGET_DEVICE) {
    // Device target means keep the output allocation cached. No copy-in needed.
    return fnaccGetCachedDeviceBuffer(static_cast<void *>(hostPtr), bytes,
        /*copyHostToDeviceOnMiss=*/false, slot, "write");
  }

  return fnaccMakeTemporaryDeviceBuffer(static_cast<void *>(hostPtr), bytes,
      /*copyHostToDevice=*/false, slot, "write");
}

static void fnaccCopyBackWriteBuffer(
    void *hostPtr, const FNACCDeviceArg &arg, std::size_t bytes) {
  // Conservative semantics: always copy writes back to host after launch.
  // Even target=device remains host-visible for now.
  FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, arg.ptr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: copied write slot %d device=0x%llx -> host=%p "
        "bytes=%zu target=%s\n",
        arg.slot, static_cast<unsigned long long>(arg.ptr),
        static_cast<void *>(hostPtr), bytes, fnaccPackTargetName(arg.target));
  }
}

static void fnaccReleaseDeviceArg(const FNACCDeviceArg &arg) {
  if (!arg.ptr)
    return;

  if (arg.cached)
    return;

  FNACC_CUDA_CHECK(cuMemFree(arg.ptr));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: freed temporary device buffer for slot %d "
        "device=0x%llx\n",
        arg.slot, static_cast<unsigned long long>(arg.ptr));
  }
}

// -------------------------------------------------------------------------- //
// CUDA module/function management
// -------------------------------------------------------------------------- //
static void fnaccDebugFunctionAttributes(CUfunction fn, int32_t kernelId) {
  if (!fnaccDebugEnabled())
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
      "FNACC: function attrs for kernel id %d: "
      "max_threads_per_block=%d num_regs=%d shared_bytes=%d "
      "binary_version=%d ptx_version=%d\n",
      kernelId, maxThreadsPerBlock, numRegs, sharedBytes, binaryVersion,
      ptxVersion);
}

static CUfunction getKernelFunction(int32_t kernelId) {
  fnaccEnsureCurrentContext();

  auto cacheIt = fnaccRegistry.functionCache.find(kernelId);
  if (cacheIt != fnaccRegistry.functionCache.end())
    return cacheIt->second;

  std::string kernelName;

  if (const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId)) {
    kernelName = desc->name;
  } else {
    kernelName = "fnacc_kernel_" + std::to_string(kernelId);

    std::fprintf(stderr,
        "FNACC warning: no JSON descriptor for kernel id %d; "
        "falling back to symbol name '%s'\n",
        kernelId, kernelName.c_str());
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: looking up CUDA kernel id %d as symbol '%s'\n",
        kernelId, kernelName.c_str());
  }

  CUfunction fn = nullptr;
  FNACC_CUDA_CHECK(
      cuModuleGetFunction(&fn, fnaccRegistry.module, kernelName.c_str()));

  fnaccRegistry.functionCache[kernelId] = fn;
  return fn;
}

static unsigned fnaccCudaThreadsPerCTA(int32_t kernelId) {
  if (const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId)) {
    if (desc->cudaThreadsPerCTA > 0)
      return static_cast<unsigned>(desc->cudaThreadsPerCTA);

    if (desc->numWarps > 0 && desc->threadsPerWarp > 0)
      return static_cast<unsigned>(desc->numWarps * desc->threadsPerWarp);
  }

  return 32;
}

static void fnaccValidateHostLaunchAgainstDesc(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ) {
  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc)
    return;

  if (desc->rank != rank) {
    std::fprintf(stderr,
        "FNACC warning: host launch rank %d disagrees with JSON "
        "rank %d for kernel id %d\n",
        rank, desc->rank, kernelId);
  }

  if (desc->tileX != blockX || desc->tileY != blockY || desc->tileZ != blockZ) {
    std::fprintf(stderr,
        "FNACC warning: host tile (%d,%d,%d) disagrees with JSON "
        "tile (%d,%d,%d) for kernel id %d\n",
        blockX, blockY, blockZ, desc->tileX, desc->tileY, desc->tileZ,
        kernelId);
  }
}

static unsigned fnaccCdiv(int32_t x, int32_t y) {
  if (y <= 0) {
    std::fprintf(stderr, "FNACC error: cdiv divisor is non-positive: %d\n", y);
    std::abort();
  }

  if (x <= 0)
    return 0;

  return static_cast<unsigned>((x + y - 1) / y);
}

static std::size_t fnaccElementCount(
    int32_t rank, int32_t extentX, int32_t extentY, int32_t extentZ) {
  std::size_t count = static_cast<std::size_t>(extentX);

  if (rank >= 2)
    count *= static_cast<std::size_t>(extentY);

  if (rank >= 3)
    count *= static_cast<std::size_t>(extentZ);

  return count;
}

static void fnaccValidateCommonLaunchInputs(const char *entryName, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, int32_t extentX, int32_t extentY, int32_t extentZ) {
  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid tile/block shape (%d,%d,%d) in %s\n", blockX,
        blockY, blockZ, entryName);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(
        stderr, "FNACC error: unsupported rank %d in %s\n", rank, entryName);
    std::abort();
  }

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "FNACC error: null host pointer in %s: "
        "a=%p b=%p c=%p\n",
        entryName, static_cast<void *>(a), static_cast<void *>(b),
        static_cast<void *>(c));
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: non-positive extent (%d,%d,%d) in %s; "
          "skipping launch\n",
          extentX, extentY, extentZ, entryName);
    }
  }
}

} // namespace

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
static std::size_t fnaccElementCountFromExtents(
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2) {
  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "FNACC error: descriptor data directive received unsupported rank %d\n",
        rank);
    std::abort();
  }

  if (extent0 < 0 || extent1 < 0 || extent2 < 0) {
    std::fprintf(stderr,
        "FNACC error: descriptor data directive received negative extent "
        "(%lld,%lld,%lld)\n",
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2));
    std::abort();
  }

  std::size_t count = static_cast<std::size_t>(extent0);

  if (rank >= 2)
    count *= static_cast<std::size_t>(extent1);

  if (rank >= 3)
    count *= static_cast<std::size_t>(extent2);

  return count;
}

static std::size_t fnaccBytesFromDescriptor(int64_t elementBytes, int32_t rank,
    int64_t extent0, int64_t extent1, int64_t extent2) {
  if (elementBytes <= 0) {
    std::fprintf(stderr,
        "FNACC error: descriptor data directive received invalid element size "
        "%lld\n",
        static_cast<long long>(elementBytes));
    std::abort();
  }

  std::size_t elements =
      fnaccElementCountFromExtents(rank, extent0, extent1, extent2);

  return fnaccCheckedMul(
      elements, static_cast<std::size_t>(elementBytes), "descriptor bytes");
}

static FNACCDeviceAllocation &fnaccGetOrCreateCachedAllocation(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnCreateOrResize,
    const char *operationName) {
  auto it = fnaccRegistry.deviceCache.find(hostPtr);

  if (it != fnaccRegistry.deviceCache.end()) {
    if (it->second.bytes == bytes)
      return it->second;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: %s resizing cached allocation for host=%p "
          "old_bytes=%zu new_bytes=%zu\n",
          operationName, hostPtr, it->second.bytes, bytes);
    }

    FNACC_CUDA_CHECK(cuMemFree(it->second.ptr));
    fnaccRegistry.deviceCache.erase(it);
  }

  FNACCDeviceAllocation allocation;
  allocation.bytes = bytes;

  if (bytes > 0) {
    FNACC_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnCreateOrResize)
      FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));
  }

  auto inserted = fnaccRegistry.deviceCache.emplace(hostPtr, allocation);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s created cached allocation host=%p device=0x%llx "
        "bytes=%zu copy_in=%s\n",
        operationName, hostPtr,
        static_cast<unsigned long long>(inserted.first->second.ptr), bytes,
        copyHostToDeviceOnCreateOrResize ? "yes" : "no");
  }

  return inserted.first->second;
}

extern "C" void __fnacc_create_bytes(void *hostPtr, int64_t bytesValue) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: create_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: create_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: create_bytes ignored zero-size object host=%p\n", hostPtr);
    }
    return;
  }

  FNACCDeviceAllocation &allocation = fnaccGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "create_bytes");

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: create_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(allocation.ptr), bytes);
  }
}

extern "C" void __fnacc_create_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: create_desc ignored null pointer\n");
    return;
  }

  fnaccValidateContiguousDescriptor("__fnacc_create_desc", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: create_desc ignored zero-size object host=%p\n", hostPtr);
    }
    return;
  }

  FNACCDeviceAllocation &allocation = fnaccGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "create_desc");

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: create_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __fnacc_update_device_bytes(void *hostPtr, int64_t bytesValue) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_device_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: update_device_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_device_bytes ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  FNACCDeviceAllocation &allocation = fnaccGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "update_device_bytes");

  FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_device_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(allocation.ptr), bytes);
  }
}

extern "C" void __fnacc_update_host_bytes(void *hostPtr, int64_t bytesValue) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_host_bytes ignored null pointer\n");
    return;
  }

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: update_host_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }

  std::size_t bytes = static_cast<std::size_t>(bytesValue);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_host_bytes ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_host_bytes ignored; no cached allocation for "
          "host=%p bytes=%zu\n",
          hostPtr, bytes);
    }
    return;
  }

  if (it->second.bytes < bytes) {
    std::fprintf(stderr,
        "FNACC error: update_host_bytes requested %zu bytes for host=%p, "
        "but cached allocation has only %zu bytes\n",
        bytes, hostPtr, it->second.bytes);
    std::abort();
  }

  FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, it->second.ptr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_host_bytes host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(it->second.ptr), bytes);
  }
}

extern "C" void __fnacc_update_device_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_device_desc ignored null pointer\n");
    return;
  }

  fnaccValidateContiguousDescriptor("__fnacc_update_device_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_device_desc ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  FNACCDeviceAllocation &allocation = fnaccGetOrCreateCachedAllocation(hostPtr,
      bytes, /*copyHostToDeviceOnCreateOrResize=*/false, "update_device_desc");

  FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_device_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __fnacc_update_host_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {

  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_host_desc ignored null pointer\n");
    return;
  }

  fnaccValidateContiguousDescriptor("__fnacc_update_host_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);

  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);

  if (bytes == 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_host_desc ignored zero-size object host=%p\n",
          hostPtr);
    }
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_host_desc ignored; no cached allocation for host=%p\n",
          hostPtr);
    }
    return;
  }

  if (it->second.bytes < bytes) {
    std::fprintf(stderr,
        "FNACC error: update_host_desc requested %zu bytes for host=%p, "
        "but cached allocation has only %zu bytes\n",
        bytes, hostPtr, it->second.bytes);
    std::abort();
  }

  FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, it->second.ptr, bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_host_desc host=%p device=0x%llx "
        "elem_bytes=%lld rank=%d extents=(%lld,%lld,%lld) bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        static_cast<long long>(elementBytes), rank,
        static_cast<long long>(extent0), static_cast<long long>(extent1),
        static_cast<long long>(extent2), bytes);
  }
}

extern "C" void __fnacc_release_desc(void *hostPtr) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: release_desc ignored null pointer\n");
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release_desc ignored; no cached allocation for host=%p\n",
          hostPtr);
    }
    return;
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  FNACC_CUDA_CHECK(cuMemFree(devicePtr));
  fnaccRegistry.deviceCache.erase(it);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: release_desc host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __fnacc_launch_nd_f32(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, int32_t extentX, int32_t extentY, int32_t extentZ) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1 && rank != 2) {
    std::fprintf(stderr,
        "FNACC error: __fnacc_launch_nd_f32 currently supports "
        "only rank 1 or 2, got rank %d\n",
        rank);
    std::abort();
  }

  fnaccValidateCommonLaunchInputs("__fnacc_launch_nd_f32", rank, blockX, blockY,
      blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);
  fnaccDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = fnaccCdiv(extentX, blockX);
  unsigned gridY = rank >= 2 ? fnaccCdiv(extentY, blockY) : 1;
  unsigned gridZ = 1;

  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes = elemCount * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNACC_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNACC_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNACC_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch binary kernel id=%d rank=%d "
        "grid=(%u,%u,%u) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, gridX, gridY, gridZ, blockX, blockY, blockZ, cudaBlockX,
        extentX, extentY, extentZ, numBytes);
  }

  if (rank == 1) {
    fnaccValidateSupportedHiddenPtrArgCount(kernelId);

    FNACCHiddenTritonArgs hidden;

    void *args[] = {
        &dA,
        &dB,
        &dC,
        &extentX,
        &hidden.hidden0,
        &hidden.hidden1,
    };

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: binary rank1 args: "
          "dA=0x%llx dB=0x%llx dC=0x%llx extentX=%d "
          "hidden0=0x%llx hidden1=0x%llx "
          "args={%p,%p,%p,%p,%p,%p}\n",
          static_cast<unsigned long long>(dA),
          static_cast<unsigned long long>(dB),
          static_cast<unsigned long long>(dC), extentX,
          static_cast<unsigned long long>(hidden.hidden0),
          static_cast<unsigned long long>(hidden.hidden1), args[0], args[1],
          args[2], args[3], args[4], args[5]);

      std::fprintf(stderr, "FNACC: about to cuLaunchKernel rank1\n");
      std::fflush(stderr);
    }

    fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

    FNACC_CUDA_CHECK(cuLaunchKernel(
        fn, gridX, 1, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: cuLaunchKernel rank1 returned\n");
      std::fflush(stderr);
    }

  } else {
    fnaccValidateSupportedHiddenPtrArgCount(kernelId);

    FNACCHiddenTritonArgs hidden;

    void *args[] = {
        &dA,
        &dB,
        &dC,
        &extentX,
        &extentY,
        &hidden.hidden0,
        &hidden.hidden1,
    };

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: binary rank2 args: "
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

      std::fprintf(stderr, "FNACC: about to cuLaunchKernel rank2\n");
      std::fflush(stderr);
    }

    fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

    FNACC_CUDA_CHECK(cuLaunchKernel(
        fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: cuLaunchKernel rank2 returned\n");
      std::fflush(stderr);
    }
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: about to cuCtxSynchronize\n");
    std::fflush(stderr);
  }

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuCtxSynchronize returned\n");
    std::fflush(stderr);
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: about to cuMemcpyDtoH c=%p dC=0x%llx bytes=%zu\n",
        static_cast<void *>(c), static_cast<unsigned long long>(dC), numBytes);
    std::fflush(stderr);
  }

  FNACC_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuMemcpyDtoH returned\n");
    std::fflush(stderr);
  }

  FNACC_CUDA_CHECK(cuMemFree(dA));
  FNACC_CUDA_CHECK(cuMemFree(dB));
  FNACC_CUDA_CHECK(cuMemFree(dC));
}

// -------------------------------------------------------------------------- //
// Public runtime ABI: 1-scalar f32 SAXPY-style kernels
// -------------------------------------------------------------------------- //
//
// Current SAXPY TTIR signature:
//
//   (%a: ptr<f32>, %b: ptr<f32>, %c: ptr<f32>, %alpha: f32, %n: i32)
//

extern "C" void __fnacc_launch_nd_f32_s1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, float scalar0, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
        "FNACC error: __fnacc_launch_nd_f32_s1 currently supports "
        "only rank 1, got rank %d\n",
        rank);
    std::abort();
  }

  fnaccValidateCommonLaunchInputs("__fnacc_launch_nd_f32_s1", rank, blockX,
      blockY, blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fnaccCdiv(extentX, blockX);
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes = numElems * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNACC_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNACC_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNACC_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  FNACCHiddenTritonArgs hidden;

  void *args[] = {
      &dA,
      &dB,
      &dC,
      &scalar0,
      &extentX,
      &hidden.hidden0,
      &hidden.hidden1,
  };

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch SAXPY kernel id=%d rank=%d "
        "grid=(%u,1,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
        "scalar0=%f bytes=%zu\n",
        kernelId, rank, gridX, blockX, blockY, blockZ, cudaBlockX, extentX,
        extentY, extentZ, static_cast<double>(scalar0), numBytes);
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: SAXPY args: "
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

    std::fprintf(stderr, "FNACC: about to cuLaunchKernel SAXPY\n");
    std::fflush(stderr);
  }

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  FNACC_CUDA_CHECK(cuLaunchKernel(
      fn, gridX, 1, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuLaunchKernel SAXPY returned\n");
    std::fflush(stderr);
  }

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  FNACC_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  FNACC_CUDA_CHECK(cuMemFree(dA));
  FNACC_CUDA_CHECK(cuMemFree(dB));
  FNACC_CUDA_CHECK(cuMemFree(dC));
}

extern "C" void __fnacc_launch_nd_f32_s2(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, float scalar0, float scalar1, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
        "FNACC error: __fnacc_launch_nd_f32_s2 currently supports "
        "only rank 1, got rank %d\n",
        rank);
    std::abort();
  }

  fnaccValidateCommonLaunchInputs("__fnacc_launch_nd_f32_s2", rank, blockX,
      blockY, blockZ, a, b, c, extentX, extentY, extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fnaccCdiv(extentX, blockX);
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes = numElems * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNACC_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNACC_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNACC_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNACC_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);

  FNACCHiddenTritonArgs hidden;

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

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch expr/s2 kernel id=%d rank=%d "
        "grid=(%u,1,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
        "scalar0=%f scalar1=%f bytes=%zu\n",
        kernelId, rank, gridX, blockX, blockY, blockZ, cudaBlockX, extentX,
        extentY, extentZ, static_cast<double>(scalar0),
        static_cast<double>(scalar1), numBytes);

    std::fprintf(stderr,
        "FNACC: s2 args: "
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

    std::fprintf(stderr, "FNACC: about to cuLaunchKernel s2\n");
    std::fflush(stderr);
  }

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  FNACC_CUDA_CHECK(cuLaunchKernel(
      fn, gridX, 1, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuLaunchKernel s2 returned\n");
    std::fflush(stderr);
  }

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  FNACC_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  FNACC_CUDA_CHECK(cuMemFree(dA));
  FNACC_CUDA_CHECK(cuMemFree(dB));
  FNACC_CUDA_CHECK(cuMemFree(dC));
}

// FNACC generic f32 launch ABI v1.
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
extern "C" void __fnacc_launch_f32_v1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays,
    int32_t numScalars, float *read0, float *read1, float *read2, float *write,
    float scalar0, float scalar1, float scalar2, int32_t extentX,
    int32_t extentY, int32_t extentZ) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "FNACC error: __fnacc_launch_f32_v1 requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "FNACC error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "FNACC error: unsupported rank %d in __fnacc_launch_f32_v1\n", rank);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid tile/block shape (%d,%d,%d) in "
        "__fnacc_launch_f32_v1\n",
        blockX, blockY, blockZ);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "FNACC error: null required pointer in __fnacc_launch_f32_v1: "
        "read0=%p write=%p\n",
        static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(
        stderr, "FNACC error: null read1 pointer in __fnacc_launch_f32_v1\n");
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(
        stderr, "FNACC error: null read2 pointer in __fnacc_launch_f32_v1\n");
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fnaccCdiv(extentX, blockX);
  unsigned gridY = rank >= 2 ? fnaccCdiv(extentY, blockY) : 1;
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes = elemCount * sizeof(float);

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for generic f32 kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "FNACC error: generic f32 launcher called for matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  // Current JSON parameter order is:
  //   slot 0 = read0
  //   slot 1 = read1
  //   slot 2 = read2 if present, otherwise write for current kernels
  //   slot numReadArrays = write
  //
  // For current FNACC kernels numReadArrays is normally 2, so write slot is 2.
  int32_t read0Slot = 0;
  int32_t read1Slot = 1;
  int32_t read2Slot = 2;
  int32_t writeSlot = numReadArrays;

  int32_t read0Target = fnaccEffectivePackTargetForSlot(desc, read0Slot, read0);

  int32_t read1Target = numReadArrays >= 2
      ? fnaccEffectivePackTargetForSlot(desc, read1Slot, read1)
      : FNACC_PACK_TARGET_HOST;

  int32_t read2Target = numReadArrays >= 3
      ? fnaccEffectivePackTargetForSlot(desc, read2Slot, read2)
      : FNACC_PACK_TARGET_HOST;

  int32_t writeTarget = fnaccEffectivePackTargetForSlot(desc, writeSlot, write);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: pack targets for kernel id %d: "
        "read0(slot %d)=%s/%s read1(slot %d)=%s/%s "
        "read2(slot %d)=%s/%s write(slot %d)=%s/%s\n",
        kernelId, read0Slot, fnaccPackTargetName(read0Target),
        fnaccPackTargetSourceName(desc, read0Slot, read0), read1Slot,
        fnaccPackTargetName(read1Target),
        numReadArrays >= 2 ? fnaccPackTargetSourceName(desc, read1Slot, read1)
                           : "unused",
        read2Slot, fnaccPackTargetName(read2Target),
        numReadArrays >= 3 ? fnaccPackTargetSourceName(desc, read2Slot, read2)
                           : "unused",
        writeSlot, fnaccPackTargetName(writeTarget),
        fnaccPackTargetSourceName(desc, writeSlot, write));
  }

  FNACCDeviceArg read0Dev =
      fnaccPrepareReadBuffer(read0, numBytes, read0Target, read0Slot);

  FNACCDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = fnaccPrepareReadBuffer(read1, numBytes, read1Target, read1Slot);

  FNACCDeviceArg read2Dev;
  if (numReadArrays >= 3)
    read2Dev = fnaccPrepareReadBuffer(read2, numBytes, read2Target, read2Slot);

  FNACCDeviceArg writeDev =
      fnaccPrepareWriteBuffer(write, numBytes, writeTarget, writeSlot);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  CUdeviceptr dRead2 = read2Dev.ptr;
  CUdeviceptr dWrite = writeDev.ptr;

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

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
        "FNACC error: internal runtime argument buffer overflow "
        "for kernel id %d; argCount=%d\n",
        kernelId, argCount);
    std::abort();
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch generic f32 kernel id=%d rank=%d "
        "reads=%d scalars=%d grid=(%u,%u,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, numReadArrays, numScalars, gridX, gridY, blockX, blockY,
        blockZ, cudaBlockX, extentX, extentY, extentZ, numBytes);
  }

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  FNACC_CUDA_CHECK(cuLaunchKernel(
      fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  if (writeDev.target == FNACC_PACK_TARGET_HOST) {
    fnaccCopyBackWriteBuffer(write, writeDev, numBytes);
  } else {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipped automatic copy-back for write slot %d "
          "because target=device; use !$fnacc update host(...) to copy back\n",
          writeDev.slot);
    }
  }

  fnaccReleaseDeviceArg(read0Dev);

  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);

  if (numReadArrays >= 3)
    fnaccReleaseDeviceArg(read2Dev);

  fnaccReleaseDeviceArg(writeDev);
}

// FNACC generic f64 launch ABI v1.
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
extern "C" void __fnacc_launch_f64_v1(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays,
    int32_t numScalars, double *read0, double *read1, double *read2,
    double *write, double scalar0, double scalar1, double scalar2,
    int32_t extentX, int32_t extentY, int32_t extentZ) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "FNACC error: __fnacc_launch_f64_v1 requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "FNACC error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(stderr,
        "FNACC error: unsupported rank %d in __fnacc_launch_f64_v1\n", rank);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid tile/block shape (%d,%d,%d) in "
        "__fnacc_launch_f64_v1\n",
        blockX, blockY, blockZ);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "FNACC error: null required pointer in __fnacc_launch_f64_v1: "
        "read0=%p write=%p\n",
        static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(
        stderr, "FNACC error: null read1 pointer in __fnacc_launch_f64_v1\n");
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(
        stderr, "FNACC error: null read2 pointer in __fnacc_launch_f64_v1\n");
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fnaccCdiv(extentX, blockX);
  unsigned gridY = rank >= 2 ? fnaccCdiv(extentY, blockY) : 1;
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);
  std::size_t numBytes = elemCount * sizeof(double);

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for generic f64 kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "FNACC error: generic f64 launcher called for matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  int32_t read0Slot = 0;
  int32_t read1Slot = 1;
  int32_t read2Slot = 2;
  int32_t writeSlot = numReadArrays;

  int32_t read0Target = fnaccEffectivePackTargetForSlot(desc, read0Slot, read0);

  int32_t read1Target = numReadArrays >= 2
      ? fnaccEffectivePackTargetForSlot(desc, read1Slot, read1)
      : FNACC_PACK_TARGET_HOST;

  int32_t read2Target = numReadArrays >= 3
      ? fnaccEffectivePackTargetForSlot(desc, read2Slot, read2)
      : FNACC_PACK_TARGET_HOST;

  int32_t writeTarget = fnaccEffectivePackTargetForSlot(desc, writeSlot, write);

  FNACCDeviceArg read0Dev =
      fnaccPrepareReadBuffer(read0, numBytes, read0Target, read0Slot);

  FNACCDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = fnaccPrepareReadBuffer(read1, numBytes, read1Target, read1Slot);

  FNACCDeviceArg read2Dev;
  if (numReadArrays >= 3)
    read2Dev = fnaccPrepareReadBuffer(read2, numBytes, read2Target, read2Slot);

  FNACCDeviceArg writeDev =
      fnaccPrepareWriteBuffer(write, numBytes, writeTarget, writeSlot);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  CUdeviceptr dRead2 = read2Dev.ptr;
  CUdeviceptr dWrite = writeDev.ptr;

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

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
        "FNACC error: internal runtime argument buffer overflow "
        "for kernel id %d; argCount=%d\n",
        kernelId, argCount);
    std::abort();
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch generic f64 kernel id=%d rank=%d "
        "reads=%d scalars=%d grid=(%u,%u,1) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
        kernelId, rank, numReadArrays, numScalars, gridX, gridY, blockX, blockY,
        blockZ, cudaBlockX, extentX, extentY, extentZ, numBytes);
  }

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  FNACC_CUDA_CHECK(cuLaunchKernel(
      fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0, nullptr, args, nullptr));

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  if (writeDev.target == FNACC_PACK_TARGET_HOST) {
    fnaccCopyBackWriteBuffer(write, writeDev, numBytes);
  } else {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipped automatic copy-back for write slot %d "
          "because target=device; use !$fnacc update host(...) to copy back\n",
          writeDev.slot);
    }
  }

  fnaccReleaseDeviceArg(read0Dev);

  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);

  if (numReadArrays >= 3)
    fnaccReleaseDeviceArg(read2Dev);

  fnaccReleaseDeviceArg(writeDev);
}

extern "C" void __fnacc_launch_matmul_f32_v1(int32_t kernelId, int32_t blockX,
    int32_t blockY, int32_t blockK, float *a, float *b, float *c, int32_t n,
    int32_t m, int32_t k) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, 2, blockX, blockY, blockK);

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "FNACC error: null host pointer in __fnacc_launch_matmul_f32_v1: "
        "a=%p b=%p c=%p\n",
        static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c));
    std::abort();
  }

  if (n <= 0 || m <= 0 || k <= 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipping matmul with non-positive extent n=%d m=%d k=%d\n", n,
          m, k);
    }
    return;
  }

  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr, "FNACC error: invalid matmul tile shape (%d,%d,%d)\n",
        blockX, blockY, blockK);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  fnaccDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = fnaccCdiv(n, blockX);
  unsigned gridY = fnaccCdiv(m, blockY);
  unsigned gridZ = 1;

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: matmul grid debug: "
        "n=%d m=%d blockX=%d blockY=%d "
        "gridX=%u gridY=%u gridZ=%u\n",
        n, m, blockX, blockY, gridX, gridY, gridZ);
  }

  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for matmul kernel id %d\n", kernelId);
    std::abort();
  }

  if (desc->kind != "matmul2d") {
    std::fprintf(stderr,
        "FNACC error: matmul launcher called for kernel id %d "
        "but JSON kind is '%s'\n",
        kernelId, desc->kind.c_str());
    std::abort();
  }

  unsigned dynamicSharedBytes =
      fnaccMatmulDynamicSharedBytes(desc, blockX, blockY, blockK);

  std::size_t bytesA =
      fnaccCheckedBytes2D(n, k, sizeof(float), "matmul A bytes");
  std::size_t bytesB =
      fnaccCheckedBytes2D(k, m, sizeof(float), "matmul B bytes");
  std::size_t bytesC =
      fnaccCheckedBytes2D(n, m, sizeof(float), "matmul C bytes");

  int32_t aTarget = fnaccEffectivePackTargetForSlot(desc, 0, a);
  int32_t bTarget = fnaccEffectivePackTargetForSlot(desc, 1, b);
  int32_t cTarget = fnaccEffectivePackTargetForSlot(desc, 2, c);

  FNACCDeviceArg aDev = fnaccPrepareReadBuffer(a, bytesA, aTarget, 0);
  FNACCDeviceArg bDev = fnaccPrepareReadBuffer(b, bytesB, bTarget, 1);
  FNACCDeviceArg cDev = fnaccPrepareWriteBuffer(c, bytesC, cTarget, 2);

  CUdeviceptr dA = aDev.ptr;
  CUdeviceptr dB = bDev.ptr;
  CUdeviceptr dC = cDev.ptr;

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

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

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch matmul kernel id=%d "
        "grid=(%u,%u,%u) grid_policy=cdiv tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) dynamic_shared_bytes=%u "
        "n=%d m=%d k=%d "
        "bytesA=%zu bytesB=%zu bytesC=%zu "
        "targets=(%s,%s,%s)\n",
        kernelId, gridX, gridY, gridZ, blockX, blockY, blockK, cudaBlockX,
        dynamicSharedBytes, n, m, k, bytesA, bytesB, bytesC,
        fnaccPackTargetName(aTarget), fnaccPackTargetName(bTarget),
        fnaccPackTargetName(cTarget));

    std::fprintf(stderr,
        "FNACC: matmul args: "
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

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);
  fnaccConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, gridZ, cudaBlockX, 1, 1,
      dynamicSharedBytes, nullptr, args, nullptr));

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  if (cDev.target == FNACC_PACK_TARGET_HOST) {
    fnaccCopyBackWriteBuffer(c, cDev, bytesC);
  } else {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipped automatic copy-back for matmul write slot %d "
          "because target=device; use !$fnacc update host(...) to copy back\n",
          cDev.slot);
    }
  }

  fnaccReleaseDeviceArg(aDev);
  fnaccReleaseDeviceArg(bDev);
  fnaccReleaseDeviceArg(cDev);
}

extern "C" void __fnacc_launch_matmul_f64_v1(int32_t kernelId, int32_t blockX,
    int32_t blockY, int32_t blockK, double *a, double *b, double *c, int32_t n,
    int32_t m, int32_t k) {
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, 2, blockX, blockY, blockK);

  if (!a || !b || !c) {
    std::fprintf(stderr,
        "FNACC error: null host pointer in __fnacc_launch_matmul_f64_v1: "
        "a=%p b=%p c=%p\n",
        static_cast<void *>(a), static_cast<void *>(b), static_cast<void *>(c));
    std::abort();
  }

  if (n <= 0 || m <= 0 || k <= 0) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipping f64 matmul with non-positive extent "
          "n=%d m=%d k=%d\n",
          n, m, k);
    }
    return;
  }

  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid f64 matmul tile shape (%d,%d,%d)\n", blockX,
        blockY, blockK);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  fnaccDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = fnaccCdiv(n, blockX);
  unsigned gridY = fnaccCdiv(m, blockY);
  unsigned gridZ = 1;

  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);

  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for f64 matmul kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (desc->kind != "matmul2d") {
    std::fprintf(stderr,
        "FNACC error: f64 matmul launcher called for kernel id %d "
        "but JSON kind is '%s'\n",
        kernelId, desc->kind.c_str());
    std::abort();
  }

  std::size_t bytesA =
      fnaccCheckedBytes2D(n, k, sizeof(double), "f64 matmul A bytes");
  std::size_t bytesB =
      fnaccCheckedBytes2D(k, m, sizeof(double), "f64 matmul B bytes");
  std::size_t bytesC =
      fnaccCheckedBytes2D(n, m, sizeof(double), "f64 matmul C bytes");

  int32_t aTarget = fnaccEffectivePackTargetForSlot(desc, 0, a);
  int32_t bTarget = fnaccEffectivePackTargetForSlot(desc, 1, b);
  int32_t cTarget = fnaccEffectivePackTargetForSlot(desc, 2, c);

  FNACCDeviceArg aDev = fnaccPrepareReadBuffer(a, bytesA, aTarget, 0);
  FNACCDeviceArg bDev = fnaccPrepareReadBuffer(b, bytesB, bTarget, 1);
  FNACCDeviceArg cDev = fnaccPrepareWriteBuffer(c, bytesC, cTarget, 2);

  CUdeviceptr dA = aDev.ptr;
  CUdeviceptr dB = bDev.ptr;
  CUdeviceptr dC = cDev.ptr;

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

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

  // The f64 fallback TTIR path does not use Triton dot or dynamic shared
  // memory.
  unsigned dynamicSharedBytes = 0;

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch f64 matmul kernel id=%d "
        "grid=(%u,%u,%u) tile=(%d,%d,%d) "
        "cuda_block=(%u,1,1) dynamic_shared_bytes=%u "
        "n=%d m=%d k=%d "
        "bytesA=%zu bytesB=%zu bytesC=%zu "
        "targets=(%s,%s,%s)\n",
        kernelId, gridX, gridY, gridZ, blockX, blockY, blockK, cudaBlockX,
        dynamicSharedBytes, n, m, k, bytesA, bytesB, bytesC,
        fnaccPackTargetName(aTarget), fnaccPackTargetName(bTarget),
        fnaccPackTargetName(cTarget));

    std::fprintf(stderr,
        "FNACC: f64 matmul args: "
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

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, gridZ, cudaBlockX, 1, 1,
      dynamicSharedBytes, nullptr, args, nullptr));

  FNACC_CUDA_CHECK(cuCtxSynchronize());

  if (cDev.target == FNACC_PACK_TARGET_HOST) {
    fnaccCopyBackWriteBuffer(c, cDev, bytesC);
  } else {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: skipped automatic copy-back for f64 matmul write slot %d "
          "because target=device; use !$fnacc update host(...) to copy back\n",
          cDev.slot);
    }
  }

  fnaccReleaseDeviceArg(aDev);
  fnaccReleaseDeviceArg(bDev);
  fnaccReleaseDeviceArg(cDev);
}

// Memory management functions to help with cached data and data lifetimes
extern "C" void __fnacc_update_host(void *hostPtr) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_host ignored null pointer\n");
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_host ignored; no cached allocation for %p\n", hostPtr);
    }
    return;
  }

  FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, it->second.ptr, it->second.bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: update_host host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        it->second.bytes);
  }
}

extern "C" void __fnacc_update_device(void *hostPtr) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_device ignored null pointer\n");
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: update_device ignored; no cached allocation for %p\n",
          hostPtr);
    }
    return;
  }

  FNACC_CUDA_CHECK(cuMemcpyHtoD(it->second.ptr, hostPtr, it->second.bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_device host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(it->second.ptr), it->second.bytes);
  }
}

extern "C" void __fnacc_release(void *hostPtr) {
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: release ignored null pointer\n");
    return;
  }

  auto it = fnaccRegistry.deviceCache.find(hostPtr);
  if (it == fnaccRegistry.deviceCache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release ignored; no cached allocation for %p\n", hostPtr);
    }
    return;
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  FNACC_CUDA_CHECK(cuMemFree(devicePtr));

  fnaccRegistry.deviceCache.erase(it);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: release host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __fnacc_release_all() {
  fnaccEnsureCurrentContext();

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: release_all releasing %zu cached allocations\n",
        fnaccRegistry.deviceCache.size());
  }

  for (auto &entry : fnaccRegistry.deviceCache) {
    void *hostPtr = entry.first;
    FNACCDeviceAllocation &allocation = entry.second;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release_all host=%p device=0x%llx bytes=%zu\n", hostPtr,
          static_cast<unsigned long long>(allocation.ptr), allocation.bytes);
    }

    FNACC_CUDA_CHECK(cuMemFree(allocation.ptr));
  }

  fnaccRegistry.deviceCache.clear();
}

extern "C" void __fnacc_register_embedded_kernels(const char *ptxData,
    std::size_t ptxSize, const char *jsonData, std::size_t jsonSize) {
  fnaccEmbeddedPtxData = ptxData;
  fnaccEmbeddedPtxSize = ptxSize;
  fnaccEmbeddedJsonData = jsonData;
  fnaccEmbeddedJsonSize = jsonSize;
}

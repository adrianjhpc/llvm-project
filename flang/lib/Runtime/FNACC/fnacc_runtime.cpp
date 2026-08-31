#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// This lock protects embedded registration, context selection, loaded-module
// and function caches, device allocations, and reduction workspaces. Embedded
// payload constructors can run before this translation unit's dynamic
// initialization, so construct it on first use regardless of final link order.
static std::recursive_mutex &fnaccGetRuntimeMutex() {
  static std::recursive_mutex mutex;
  return mutex;
}
#define FNACC_RUNTIME_GUARD() \
  std::lock_guard<std::recursive_mutex> fnaccRuntimeLock(fnaccGetRuntimeMutex())

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
    "FNACC_RUNTIME_BUILD_ID_nested_data_regions_v11";

static std::size_t fnaccCheckedMul(
    std::size_t a, std::size_t b, const char *what) {
  if (a != 0 && b > static_cast<std::size_t>(-1) / a) {
    std::fprintf(
        stderr, "FNACC error: size overflow while computing %s\n", what);
    std::abort();
  }

  return a * b;
}

static std::size_t fnaccCheckedAdd(
    std::size_t a, std::size_t b, const char *what) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    std::fprintf(
        stderr, "FNACC error: size overflow while computing %s\n", what);
    std::abort();
  }
  return a + b;
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

  if (extent0 < 0 || extent1 < 0 || extent2 < 0) {
    std::fprintf(
        stderr, "FNACC error: %s received a negative extent\n", operationName);
    std::abort();
  }

  std::size_t expected0Size = static_cast<std::size_t>(elementBytes);
  std::size_t expected1Size = fnaccCheckedMul(expected0Size,
      static_cast<std::size_t>(extent0), "descriptor byte stride 1");
  std::size_t expected2Size = fnaccCheckedMul(expected1Size,
      static_cast<std::size_t>(extent1), "descriptor byte stride 2");

  if (expected2Size >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    std::fprintf(stderr, "FNACC error: %s descriptor stride exceeds i64\n",
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

enum class FNACCKernelParameterRole {
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

struct FNACCKernelParameterDesc {
  int32_t slot = -1;
  FNACCKernelParameterRole role = FNACCKernelParameterRole::Unknown;
  std::string type;
  int32_t arrayIndex = -1;
  int32_t scalarIndex = -1;
  int32_t dimension = -1;
};

static FNACCKernelParameterRole fnaccParseParameterRole(
    const std::string &role) {
  if (role == "read")
    return FNACCKernelParameterRole::Read;
  if (role == "write")
    return FNACCKernelParameterRole::Write;
  if (role == "read_write")
    return FNACCKernelParameterRole::ReadWrite;
  if (role == "partials")
    return FNACCKernelParameterRole::Partials;
  if (role == "scalar")
    return FNACCKernelParameterRole::Scalar;
  if (role == "extent_x")
    return FNACCKernelParameterRole::ExtentX;
  if (role == "extent_y")
    return FNACCKernelParameterRole::ExtentY;
  if (role == "extent_k" || role == "extent_z")
    return FNACCKernelParameterRole::ExtentZ;
  if (role == "loop_lower_x")
    return FNACCKernelParameterRole::LoopLowerX;
  if (role == "loop_lower_y")
    return FNACCKernelParameterRole::LoopLowerY;
  if (role == "loop_lower_z")
    return FNACCKernelParameterRole::LoopLowerZ;
  if (role == "array_lower_bound")
    return FNACCKernelParameterRole::ArrayLowerBound;
  if (role == "array_stride")
    return FNACCKernelParameterRole::ArrayStride;
  return FNACCKernelParameterRole::Unknown;
}

static std::vector<FNACCKernelParameterDesc> jsonParseParameterEntries(
    const std::string &kernelObjectText) {
  std::vector<FNACCKernelParameterDesc> parameters;
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
    FNACCKernelParameterDesc parameter;
    std::string role;
    if (!jsonFindInt(objectText, "slot", parameter.slot) ||
        !jsonFindString(objectText, "role", role) ||
        !jsonFindString(objectText, "type", parameter.type)) {
      pos = objectEnd;
      continue;
    }
    parameter.role = fnaccParseParameterRole(role);
    jsonFindInt(objectText, "array_index", parameter.arrayIndex);
    jsonFindInt(objectText, "scalar_index", parameter.scalarIndex);
    jsonFindInt(objectText, "dimension", parameter.dimension);
    // Stencil metadata emitted before the Stage 1 migration ordered scalars
    // but did not spell out scalar_index. Preserve compatibility with those
    // launch-ABI-v2 objects while making new metadata explicit.
    if (parameter.role == FNACCKernelParameterRole::Scalar) {
      if (parameter.scalarIndex < 0)
        parameter.scalarIndex = nextImplicitScalarIndex;
      nextImplicitScalarIndex =
          std::max(nextImplicitScalarIndex, parameter.scalarIndex + 1);
    }
    parameters.push_back(std::move(parameter));
    pos = objectEnd;
  }

  std::sort(parameters.begin(), parameters.end(),
      [](const FNACCKernelParameterDesc &lhs,
          const FNACCKernelParameterDesc &rhs) { return lhs.slot < rhs.slot; });
  return parameters;
}

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

  std::string backend = "triton";
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
  std::vector<FNACCPackEntry> pack;

  // Ordered, source-visible device parameters used by launch ABI v2.
  std::vector<FNACCKernelParameterDesc> parameters;

  int32_t ptxIndex = 0;
  std::string ptxFile;
};

static std::size_t fnaccScalarParameterBytes(const std::string &type) {
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

static bool fnaccIsPointerParameterType(const std::string &type) {
  return type.size() > 5 && type.compare(0, 4, "ptr<") == 0 &&
      type.back() == '>';
}

static void fnaccValidateVariadicKernelMetadata(const FNACCKernelDesc &desc) {
  if (desc.rank < 1 || desc.rank > 2 || desc.arrayCount <= 0 ||
      desc.scalarCount < 0 || desc.outputCount <= 0 ||
      desc.parameters.empty()) {
    std::fprintf(stderr,
        "FNACC error: invalid v2 ABI metadata for kernel id %d\n", desc.id);
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
        "FNACC error: invalid v2 reduction output count for kernel id %d\n",
        desc.id);
    std::abort();
  }

  for (std::size_t index = 0; index < desc.parameters.size(); ++index) {
    const FNACCKernelParameterDesc &parameter = desc.parameters[index];
    if (parameter.slot != static_cast<int32_t>(index)) {
      std::fprintf(stderr,
          "FNACC error: non-contiguous v2 parameter slots for kernel id %d\n",
          desc.id);
      std::abort();
    }

    auto validateArray = [&](bool isPointerParameter = true) {
      if (parameter.arrayIndex < 0 || parameter.arrayIndex >= desc.arrayCount) {
        std::fprintf(stderr,
            "FNACC error: invalid v2 array index for kernel id %d slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      if (isPointerParameter)
        arraysReferenced[static_cast<std::size_t>(parameter.arrayIndex)] = true;
    };

    switch (parameter.role) {
    case FNACCKernelParameterRole::Read:
    case FNACCKernelParameterRole::Write:
    case FNACCKernelParameterRole::ReadWrite:
      validateArray();
      if (!fnaccIsPointerParameterType(parameter.type)) {
        std::fprintf(stderr,
            "FNACC error: v2 array parameter is not a pointer for kernel id "
            "%d slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    case FNACCKernelParameterRole::Scalar:
      if (parameter.scalarIndex < 0 ||
          parameter.scalarIndex >= desc.scalarCount ||
          fnaccScalarParameterBytes(parameter.type) == 0) {
        std::fprintf(stderr,
            "FNACC error: invalid v2 scalar metadata for kernel id %d slot "
            "%d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      scalarsReferenced[static_cast<std::size_t>(parameter.scalarIndex)] = true;
      break;
    case FNACCKernelParameterRole::ExtentX:
    case FNACCKernelParameterRole::ExtentY:
    case FNACCKernelParameterRole::ExtentZ: {
      unsigned dim = parameter.role == FNACCKernelParameterRole::ExtentX ? 0
          : parameter.role == FNACCKernelParameterRole::ExtentY          ? 1
                                                                         : 2;
      bool validDimension =
          dim < static_cast<unsigned>(desc.rank) || (isMatmul && dim == 2);
      if (!validDimension || parameter.type != "i32") {
        std::fprintf(stderr,
            "FNACC error: invalid v2 extent metadata for kernel id %d slot "
            "%d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      ++extentCounts[dim];
      break;
    }
    case FNACCKernelParameterRole::LoopLowerX:
    case FNACCKernelParameterRole::LoopLowerY:
    case FNACCKernelParameterRole::LoopLowerZ: {
      unsigned dim = parameter.role == FNACCKernelParameterRole::LoopLowerX ? 0
          : parameter.role == FNACCKernelParameterRole::LoopLowerY          ? 1
                                                                            : 2;
      if (dim >= static_cast<unsigned>(desc.rank) || parameter.type != "i32") {
        std::fprintf(stderr,
            "FNACC error: invalid v2 loop-lower metadata for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    }
    case FNACCKernelParameterRole::ArrayLowerBound:
    case FNACCKernelParameterRole::ArrayStride:
      validateArray(false);
      if (parameter.dimension < 0 || parameter.dimension >= desc.rank ||
          parameter.type != "i32") {
        std::fprintf(stderr,
            "FNACC error: invalid v2 array-layout metadata for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      break;
    case FNACCKernelParameterRole::Partials:
      if (!isReduction || !fnaccIsPointerParameterType(parameter.type)) {
        std::fprintf(stderr,
            "FNACC error: invalid v2 partials parameter for kernel id %d "
            "slot %d\n",
            desc.id, parameter.slot);
        std::abort();
      }
      ++partialsCount;
      break;
    case FNACCKernelParameterRole::Unknown:
      std::fprintf(stderr,
          "FNACC error: unsupported v2 parameter role for kernel id %d slot "
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
        "FNACC error: incomplete v2 parameter metadata for kernel id %d\n",
        desc.id);
    std::abort();
  }
}

struct FNACCDeviceAllocation {
  CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
  std::size_t dataRegionReferences = 0;
};

struct FNACCDataRegionFrame {
  std::vector<void *> allocations;
};

struct FNACCReductionBufferStats {
  // Number of cuMemAlloc calls. Growth allocations are included here and are
  // also counted separately below.
  uint64_t allocations = 0;
  uint64_t growths = 0;
  uint64_t reuses = 0;
};

struct FNACCReductionWorkspace {
  CUdevice device = 0;
  CUcontext context = nullptr;

  // The primary kernel writes one partial per Triton program to this buffer.
  FNACCDeviceAllocation partials;
  FNACCReductionBufferStats partialStats;

  // Hierarchical stages ping-pong between partials and this buffer.
  FNACCDeviceAllocation scratch;
  FNACCReductionBufferStats scratchStats;

  uint64_t primaryLaunches = 0;
  uint64_t stageLaunches = 0;
};

struct FNACCContextState {
  CUdevice device = 0;
  CUcontext context = nullptr;
  bool retainedPrimaryContext = false;
  CUstream stream = nullptr;
  CUevent completionEvent = nullptr;
  std::vector<CUmodule> modules;
  std::unordered_map<int32_t, CUfunction> functionCache;
  std::unordered_map<void *, FNACCDeviceAllocation> deviceCache;
  std::vector<FNACCDataRegionFrame> dataRegions;
  FNACCReductionWorkspace reductionWorkspace;
};

struct FNACCKernelRegistry {
  bool initialized = false;

  std::unordered_map<int32_t, FNACCKernelDesc> kernels;
  std::vector<std::string> ptxTexts;

  // CUDA modules, functions, allocations, streams and reduction buffers are
  // all context-owned. Never reuse any of them in a different context, even
  // when two contexts select the same CUDA device.
  std::unordered_map<CUcontext, FNACCContextState> contexts;
  std::unordered_map<int, CUcontext> primaryContexts;
  CUcontext activeContext = nullptr;
};

struct FNACCEmbeddedKernelBundle {
  // Keep these integer values in sync with the generated C bundle ABI.
  enum ImageKind : int32_t { PTX = 1, Cubin = 2 };

  std::vector<const void *> imageData;
  std::vector<std::size_t> imageSize;
  std::vector<int32_t> imageKind;
  const char *jsonData = nullptr;
  std::size_t jsonSize = 0;
};

// The generated bundle is registered from a constructor in another
// translation unit.  A function-local static prevents that constructor from
// writing into namespace-scope vectors before their constructors have run.
static std::vector<FNACCEmbeddedKernelBundle> &fnaccGetEmbeddedKernelBundles() {
  static std::vector<FNACCEmbeddedKernelBundle> bundles;
  return bundles;
}

static FNACCKernelRegistry fnaccRegistry;

static FNACCContextState &fnaccActiveContextState() {
  auto it = fnaccRegistry.contexts.find(fnaccRegistry.activeContext);
  if (it == fnaccRegistry.contexts.end()) {
    std::fprintf(stderr, "FNACC error: no active CUDA context state\n");
    std::abort();
  }
  return it->second;
}

struct FNACCDeviceArg {
  CUdeviceptr ptr = 0;
  bool cached = false;
  int32_t target = FNACC_PACK_TARGET_HOST;
  int32_t slot = -1;
};

static bool fnaccEnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool fnaccDebugEnabled() { return fnaccEnvFlagEnabled("FNACC_DEBUG"); }

static bool fnaccReductionStatsEnabled() {
  return fnaccEnvFlagEnabled("FNACC_REDUCTION_STATS");
}

class FNACCCurrentContextGuard {
public:
  FNACCCurrentContextGuard() {
    CUresult result = cuCtxGetCurrent(&previousContext);

    if (result == CUDA_ERROR_NOT_INITIALIZED) {
      previousContext = nullptr;
      return;
    }

    FNACC_CUDA_CHECK(result);
  }

  ~FNACCCurrentContextGuard() {
    CUresult result = cuCtxSetCurrent(previousContext);

    if (result == CUDA_ERROR_NOT_INITIALIZED)
      return;

    if (result != CUDA_SUCCESS && fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC warning: failed to restore the caller's CUDA context\n");
    }
  }

private:
  CUcontext previousContext{nullptr};
};

static FNACCReductionWorkspace fnaccAggregateReductionWorkspaceStats() {
  FNACCReductionWorkspace total;
  for (const auto &entry : fnaccRegistry.contexts) {
    const FNACCReductionWorkspace &workspace = entry.second.reductionWorkspace;
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

static void fnaccPrintReductionWorkspaceStats() {
  FNACCReductionWorkspace workspace = fnaccAggregateReductionWorkspaceStats();
  std::fprintf(stderr,
      "FNACC reduction workspace: primary_launches=%llu "
      "stage_launches=%llu "
      "contexts=%zu "
      "partials={allocations=%llu,growths=%llu,reuses=%llu,capacity_bytes=%zu} "
      "scratch={allocations=%llu,growths=%llu,reuses=%llu,capacity_bytes=%zu}"
      "\n",
      static_cast<unsigned long long>(workspace.primaryLaunches),
      static_cast<unsigned long long>(workspace.stageLaunches),
      fnaccRegistry.contexts.size(),
      static_cast<unsigned long long>(workspace.partialStats.allocations),
      static_cast<unsigned long long>(workspace.partialStats.growths),
      static_cast<unsigned long long>(workspace.partialStats.reuses),
      workspace.partials.bytes,
      static_cast<unsigned long long>(workspace.scratchStats.allocations),
      static_cast<unsigned long long>(workspace.scratchStats.growths),
      static_cast<unsigned long long>(workspace.scratchStats.reuses),
      workspace.scratch.bytes);
}

static const char *fnaccEmbeddedImageKindName(int32_t kind) {
  switch (kind) {
  case FNACCEmbeddedKernelBundle::PTX:
    return "ptx";
  case FNACCEmbeddedKernelBundle::Cubin:
    return "cubin";
  default:
    return "unknown";
  }
}

static bool fnaccHasEmbeddedBundles() {
  const auto &bundles = fnaccGetEmbeddedKernelBundles();
  if (bundles.empty())
    return false;
  for (const FNACCEmbeddedKernelBundle &bundle : bundles) {
    if (!bundle.jsonData || bundle.jsonSize == 0 || bundle.imageData.empty() ||
        bundle.imageData.size() != bundle.imageSize.size() ||
        bundle.imageData.size() != bundle.imageKind.size())
      return false;
    for (std::size_t i = 0; i < bundle.imageData.size(); ++i) {
      if (!bundle.imageData[i] || bundle.imageSize[i] == 0 ||
          (bundle.imageKind[i] != FNACCEmbeddedKernelBundle::PTX &&
              bundle.imageKind[i] != FNACCEmbeddedKernelBundle::Cubin))
        return false;
    }
  }
  return true;
}

static std::vector<std::string> fnaccGetImagesFromEmbeddedBundles() {
  std::vector<std::string> result;

  if (!fnaccHasEmbeddedBundles())
    return result;

  const auto &bundles = fnaccGetEmbeddedKernelBundles();
  for (std::size_t bundleIndex = 0; bundleIndex < bundles.size();
      ++bundleIndex) {
    const FNACCEmbeddedKernelBundle &bundle = bundles[bundleIndex];
    for (std::size_t i = 0; i < bundle.imageData.size(); ++i) {
      const char *bytes = static_cast<const char *>(bundle.imageData[i]);
      result.emplace_back(bytes, bundle.imageSize[i]);
      if (fnaccDebugEnabled()) {
        std::fprintf(stderr,
            "FNACC: loading embedded %s bundle=%zu entry=%zu bytes=%zu\n",
            fnaccEmbeddedImageKindName(bundle.imageKind[i]), bundleIndex, i,
            bundle.imageSize[i]);
      }
    }
  }
  return result;
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

static std::vector<std::string> fnaccGetPtxTextsFromDirectory(
    const std::unordered_map<int32_t, FNACCKernelDesc> &kernels) {
  std::vector<std::string> result;

  const char *dir = std::getenv("FNACC_PTX_DIR");
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
    const FNACCKernelDesc &desc = entry.second;

    std::string path = std::string(dir) + "/" + desc.ptxFile;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: loading PTX for kernel id %d from '%s'\n",
          desc.id, path.c_str());
    }

    result[static_cast<std::size_t>(desc.ptxIndex)] =
        fnaccReadTextFile(path.c_str());
  }

  return result;
}

static std::vector<std::string> fnaccGetPtxTexts(
    const std::unordered_map<int32_t, FNACCKernelDesc> &kernels) {
  const char *singlePtxPath = std::getenv("FNACC_PTX");

  if (singlePtxPath && singlePtxPath[0] != '\0') {
    if (fnaccDebugEnabled())
      std::fprintf(
          stderr, "FNACC: loading single PTX from '%s'\n", singlePtxPath);

    return {fnaccReadTextFile(singlePtxPath)};
  }

  if (const char *dir = std::getenv("FNACC_PTX_DIR")) {
    if (dir[0] != '\0')
      return fnaccGetPtxTextsFromDirectory(kernels);
  }

  if (fnaccHasEmbeddedBundles())
    return fnaccGetImagesFromEmbeddedBundles();

  // Backwards-compatible fallback.
  return {fnaccReadTextFile("fnacc_kernels.ptx")};
}

static std::string fnaccGetJsonText() {
  const char *jsonPath = std::getenv("FNACC_KERNELS_JSON");

  if (jsonPath && jsonPath[0] != '\0') {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: loading JSON from '%s'\n", jsonPath);

    return fnaccReadTextFile(jsonPath);
  }

  if (fnaccHasEmbeddedBundles() &&
      fnaccGetEmbeddedKernelBundles().size() == 1) {
    const FNACCEmbeddedKernelBundle &bundle =
        fnaccGetEmbeddedKernelBundles().front();
    if (fnaccDebugEnabled()) {
      std::fprintf(
          stderr, "FNACC: loading embedded JSON, bytes=%zu\n", bundle.jsonSize);
    }

    return std::string(bundle.jsonData, bundle.jsonSize);
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
    jsonFindString(objectText, "backend", desc.backend);
    jsonFindString(objectText, "device_image_kind", desc.deviceImageKind);

    if (!jsonFindInt(objectText, "image_index", desc.ptxIndex))
      jsonFindInt(objectText, "ptx_index", desc.ptxIndex);
    if (!jsonFindString(objectText, "image_file", desc.ptxFile))
      jsonFindString(objectText, "ptx_file", desc.ptxFile);

    if (desc.ptxFile.empty())
      desc.ptxFile =
          desc.name + (desc.deviceImageKind == "cubin" ? ".cubin" : ".ptx");

    if (desc.id < 0 || desc.ptxIndex < 0) {
      std::fprintf(stderr,
          "FNACC error: kernel JSON contains a negative id or image index\n");
      std::abort();
    }

    if (desc.deviceImageKind != "ptx" && desc.deviceImageKind != "cubin") {
      std::fprintf(stderr,
          "FNACC error: kernel id %d has unsupported device image kind '%s'\n",
          desc.id, desc.deviceImageKind.c_str());
      std::abort();
    }

    jsonFindInt(objectText, "rank", desc.rank);

    jsonFindIntArray3(objectText, "tile", desc.tileX, desc.tileY, desc.tileZ);

    jsonFindInt(objectText, "num_warps", desc.numWarps);
    jsonFindInt(objectText, "threads_per_warp", desc.threadsPerWarp);
    jsonFindInt(objectText, "num_ctas", desc.numCTAs);
    jsonFindInt(objectText, "num_stages", desc.numStages);

    bool hasCudaThreads =
        jsonFindInt(objectText, "cuda_threads_per_cta", desc.cudaThreadsPerCTA);

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
        desc.reductionOp = FNACCKernelDesc::ReductionOperator::Add;
      else if (reductionOp == "multiply")
        desc.reductionOp = FNACCKernelDesc::ReductionOperator::Multiply;
      else if (reductionOp == "min")
        desc.reductionOp = FNACCKernelDesc::ReductionOperator::Min;
      else if (reductionOp == "max")
        desc.reductionOp = FNACCKernelDesc::ReductionOperator::Max;
      else {
        std::fprintf(stderr,
            "FNACC error: kernel id %d has unknown reduction_op '%s'\n",
            desc.id, reductionOp.c_str());
        std::abort();
      }
    }

    int64_t expectedCudaThreads =
        static_cast<int64_t>(desc.numWarps) * desc.threadsPerWarp;
    if (desc.rank < 1 || desc.rank > 3 || desc.tileX <= 0 || desc.tileY <= 0 ||
        desc.tileZ <= 0 || desc.numWarps <= 0 || desc.threadsPerWarp != 32 ||
        desc.numCTAs <= 0 || desc.numStages <= 0 || expectedCudaThreads <= 0 ||
        expectedCudaThreads > std::numeric_limits<int32_t>::max()) {
      std::fprintf(stderr,
          "FNACC error: invalid launch metadata for kernel id %d\n", desc.id);
      std::abort();
    }

    if (!hasCudaThreads)
      desc.cudaThreadsPerCTA = static_cast<int32_t>(expectedCudaThreads);
    if (desc.cudaThreadsPerCTA != expectedCudaThreads) {
      std::fprintf(stderr,
          "FNACC error: cuda_threads_per_cta disagrees with warp metadata "
          "for kernel id %d\n",
          desc.id);
      std::abort();
    }
    if (desc.tritonHiddenPtrArgs != 2) {
      std::fprintf(stderr,
          "FNACC error: kernel id %d requires %d private pointer parameters; "
          "this runtime ABI supports exactly 2\n",
          desc.id, desc.tritonHiddenPtrArgs);
      std::abort();
    }
    if (desc.kind == "stencil2d" && desc.launchAbiVersion != 2) {
      std::fprintf(stderr,
          "FNACC error: invalid stencil2d ABI metadata for kernel id %d\n",
          desc.id);
      std::abort();
    }
    if (desc.launchAbiVersion == 2)
      fnaccValidateVariadicKernelMetadata(desc);

    for (const auto &entry : result) {
      if (entry.second.name == desc.name) {
        std::fprintf(stderr, "FNACC error: duplicate kernel name '%s'\n",
            desc.name.c_str());
        std::abort();
      }
    }
    if (!result.emplace(desc.id, std::move(desc)).second) {
      std::fprintf(stderr, "FNACC error: duplicate kernel id in JSON\n");
      std::abort();
    }

    pos = objectEnd;
  }

  return result;
}

static void fnaccCleanup() {
  FNACC_RUNTIME_GUARD();
  if (!fnaccRegistry.initialized)
    return;

  if (fnaccReductionStatsEnabled())
    fnaccPrintReductionWorkspaceStats();

  for (auto &entry : fnaccRegistry.contexts) {
    FNACCContextState &state = entry.second;
    FNACCReductionWorkspace &workspace = state.reductionWorkspace;
    if (state.context)
      cuCtxSetCurrent(state.context);
    if (workspace.partials.ptr)
      cuMemFree(workspace.partials.ptr);
    if (workspace.scratch.ptr)
      cuMemFree(workspace.scratch.ptr);

    if (state.stream)
      cuStreamSynchronize(state.stream);
    for (auto &allocation : state.deviceCache)
      if (allocation.second.ptr)
        cuMemFree(allocation.second.ptr);
    state.deviceCache.clear();
    state.dataRegions.clear();
    state.functionCache.clear();
    for (CUmodule module : state.modules)
      if (module)
        cuModuleUnload(module);
    state.modules.clear();
    if (state.completionEvent)
      cuEventDestroy(state.completionEvent);
    if (state.stream)
      cuStreamDestroy(state.stream);
  }

  for (auto &entry : fnaccRegistry.contexts)
    if (entry.second.retainedPrimaryContext)
      cuDevicePrimaryCtxRelease(entry.second.device);

  fnaccRegistry.contexts.clear();
  fnaccRegistry.primaryContexts.clear();
  fnaccRegistry.activeContext = nullptr;
  fnaccRegistry.ptxTexts.clear();
  fnaccRegistry.kernels.clear();

  fnaccRegistry.initialized = false;
}

static unsigned fnaccReductionIntegerDynamicSharedBytes(
    const FNACCKernelDesc *desc, int32_t blockX, size_t integerSize) {

  std::size_t bytes = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      integerSize, "reduction dynamic shared bytes");

  // Align to 256 bytes.
  bytes = fnaccCheckedAdd(bytes, 255, "reduction shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: reduction dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
}

static unsigned fnaccReductionDynamicSharedBytes(
    const FNACCKernelDesc *desc, int32_t blockX) {

  std::size_t bytes = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      sizeof(float), "reduction dynamic shared bytes");

  // Align to 256 bytes.
  bytes = fnaccCheckedAdd(bytes, 255, "reduction shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: reduction dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
}

static unsigned fnaccReductionF64DynamicSharedBytes(
    const FNACCKernelDesc *desc, int32_t blockX) {

  std::size_t bytes = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      sizeof(double), "reduction f64 dynamic shared bytes");

  // Align to 256 bytes.
  bytes = fnaccCheckedAdd(bytes, 255, "reduction f64 shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton may require padding/alignment beyond the simple tile estimate.
  // Keep the conservative prototype minimum for now.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: reduction f64 dynamic shared memory requirement too "
        "large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  return requiredBytes;
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

  std::size_t elems = fnaccCheckedMul(
      fnaccCheckedAdd(aElems, bElems, "matmul shared tile elements"),
      static_cast<std::size_t>(stages),
      "matmul dynamic shared staged tile elements");

  std::size_t bytes =
      fnaccCheckedMul(elems, sizeof(float), "matmul dynamic shared bytes");

  // Align to 256 bytes.
  bytes = fnaccCheckedAdd(bytes, 255, "matmul shared alignment") &
      ~static_cast<std::size_t>(255);

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

static unsigned fnaccMatmulF64DynamicSharedBytes(
    const FNACCKernelDesc * /*desc*/, int32_t blockX, int32_t blockY,
    int32_t blockK) {
  if (blockX <= 0 || blockY <= 0 || blockK <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid f64 matmul tile shape (%d,%d,%d) while "
        "computing dynamic shared memory\n",
        blockX, blockY, blockK);
    std::abort();
  }

  // The f64 blocked fallback materialises/reduces an M x N x K product-like
  // tensor and Triton lowering uses dynamic shared memory for parts of the
  // lowering. This is a conservative estimate; the optional environment
  // override below can be used to tune/debug.
  std::size_t aElems = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockK), "f64 matmul shared A tile elements");

  std::size_t bElems = fnaccCheckedMul(static_cast<std::size_t>(blockK),
      static_cast<std::size_t>(blockY), "f64 matmul shared B tile elements");

  std::size_t prodElems =
      fnaccCheckedMul(fnaccCheckedMul(static_cast<std::size_t>(blockX),
                          static_cast<std::size_t>(blockY),
                          "f64 matmul shared product M*N elements"),
          static_cast<std::size_t>(blockK),
          "f64 matmul shared product M*N*K elements");

  std::size_t accElems = fnaccCheckedMul(static_cast<std::size_t>(blockX),
      static_cast<std::size_t>(blockY), "f64 matmul shared accumulator elems");

  std::size_t elems = fnaccCheckedAdd(
      fnaccCheckedAdd(aElems, bElems, "f64 matmul A/B shared elements"),
      fnaccCheckedAdd(prodElems, accElems,
          "f64 matmul product/accumulator shared elements"),
      "f64 matmul total shared elements");

  std::size_t bytes =
      fnaccCheckedMul(elems, sizeof(double), "f64 matmul dynamic shared bytes");

  // Align to 256 bytes.
  bytes = fnaccCheckedAdd(bytes, 255, "f64 matmul shared alignment") &
      ~static_cast<std::size_t>(255);

  // Triton-generated kernels often assume a non-trivial shared-memory arena.
  // Keep a conservative minimum.
  if (bytes < 16384)
    bytes = 16384;

  if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: f64 matmul dynamic shared memory requirement too large: "
        "%zu bytes\n",
        bytes);
    std::abort();
  }

  unsigned requiredBytes = static_cast<unsigned>(bytes);

  // Debug/tuning override. Do not allow values below the computed minimum.
  if (const char *value = std::getenv("FNACC_MATMUL_F64_SHARED_BYTES")) {
    if (value[0] != '\0') {
      unsigned requested = fnaccGetEnvUnsignedAllowZero(
          "FNACC_MATMUL_F64_SHARED_BYTES", requiredBytes);

      if (requested < requiredBytes) {
        std::fprintf(stderr,
            "FNACC error: FNACC_MATMUL_F64_SHARED_BYTES=%u is smaller than "
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

static void fnaccEnsureInitialized() {
  if (fnaccRegistry.initialized)
    return;

  if (fnaccDebugEnabled())
    std::fprintf(
        stderr, "FNACC: runtime build id: %s\n", FNACC_RUNTIME_BUILD_ID);

  FNACC_CUDA_CHECK(cuInit(0));
  auto parseOneJson = [&](const std::string &json, int32_t imageBase) {
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

    auto kernels = fnaccParseKernelDescsFromJson(json);
    for (auto &entry : kernels) {
      FNACCKernelDesc &desc = entry.second;
      desc.ptxIndex += imageBase;
      for (const auto &existing : fnaccRegistry.kernels) {
        if (existing.first == desc.id || existing.second.name == desc.name) {
          std::fprintf(stderr,
              "FNACC error: embedded bundles contain colliding kernel "
              "identity id=%d name='%s'\n",
              desc.id, desc.name.c_str());
          std::abort();
        }
      }
      fnaccRegistry.kernels.emplace(entry.first, std::move(desc));
    }
  };

  const char *jsonOverride = std::getenv("FNACC_KERNELS_JSON");
  bool useEmbeddedBundles =
      (!jsonOverride || jsonOverride[0] == '\0') && fnaccHasEmbeddedBundles();
  if (useEmbeddedBundles) {
    int32_t imageBase = 0;
    for (const FNACCEmbeddedKernelBundle &bundle :
        fnaccGetEmbeddedKernelBundles()) {
      parseOneJson(std::string(bundle.jsonData, bundle.jsonSize), imageBase);
      imageBase += static_cast<int32_t>(bundle.imageData.size());
    }
  } else {
    parseOneJson(fnaccGetJsonText(), 0);
  }

  fnaccRegistry.ptxTexts = fnaccGetPtxTexts(fnaccRegistry.kernels);

  if (fnaccRegistry.ptxTexts.empty()) {
    std::fprintf(stderr, "FNACC error: no device images were available\n");
    std::abort();
  }

  for (std::size_t i = 0; i < fnaccRegistry.ptxTexts.size(); ++i) {
    if (fnaccRegistry.ptxTexts[i].empty()) {
      std::fprintf(stderr,
          "FNACC error: device image entry %zu is empty or missing\n", i);
      std::abort();
    }
  }

  if (fnaccDebugEnabled()) {
    for (const auto &entry : fnaccRegistry.kernels) {
      const FNACCKernelDesc &desc = entry.second;

      std::fprintf(stderr,
          "FNACC: registered kernel id %d -> '%s' "
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

  fnaccRegistry.initialized = true;
  std::atexit(fnaccCleanup);
}

static FNACCContextState &fnaccCreateContextState(
    CUdevice device, CUcontext context, bool retainedPrimaryContext) {
  FNACCContextState state;
  state.device = device;
  state.context = context;
  state.retainedPrimaryContext = retainedPrimaryContext;
  FNACC_CUDA_CHECK(cuCtxSetCurrent(state.context));
  FNACC_CUDA_CHECK(cuStreamCreate(&state.stream, CU_STREAM_DEFAULT));
  FNACC_CUDA_CHECK(
      cuEventCreate(&state.completionEvent, CU_EVENT_DISABLE_TIMING));

  state.modules.resize(fnaccRegistry.ptxTexts.size(), nullptr);
  for (std::size_t i = 0; i < fnaccRegistry.ptxTexts.size(); ++i) {
    FNACC_CUDA_CHECK(cuModuleLoadDataEx(&state.modules[i],
        fnaccRegistry.ptxTexts[i].c_str(), 0, nullptr, nullptr));
    if (fnaccDebugEnabled())
      std::fprintf(stderr,
          "FNACC: loaded CUDA module %zu on device=%d context=%p\n", i,
          static_cast<int>(device), static_cast<void *>(state.context));
  }

  auto inserted = fnaccRegistry.contexts.emplace(context, std::move(state));
  if (!inserted.second) {
    std::fprintf(stderr,
        "FNACC error: duplicate CUDA context state for context %p\n",
        static_cast<void *>(context));
    std::abort();
  }
  return inserted.first->second;
}

static FNACCContextState &fnaccGetOrCreatePrimaryContextState(int ordinal) {
  auto known = fnaccRegistry.primaryContexts.find(ordinal);
  if (known != fnaccRegistry.primaryContexts.end())
    return fnaccRegistry.contexts.at(known->second);

  CUdevice device = 0;
  CUcontext context = nullptr;
  FNACC_CUDA_CHECK(cuDeviceGet(&device, ordinal));
  FNACC_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));

  FNACCContextState &state =
      fnaccCreateContextState(device, context, /*retainedPrimaryContext=*/true);
  fnaccRegistry.primaryContexts.emplace(ordinal, context);
  return state;
}

static void fnaccEnsureCurrentContext() {
  fnaccEnsureInitialized();

  if (fnaccEnvFlagEnabled("FNACC_USE_CURRENT_CONTEXT")) {
    CUcontext context = nullptr;
    CUdevice device = 0;
    FNACC_CUDA_CHECK(cuCtxGetCurrent(&context));
    if (!context) {
      std::fprintf(stderr,
          "FNACC error: FNACC_USE_CURRENT_CONTEXT is set but no CUDA context "
          "is current\n");
      std::abort();
    }
    FNACC_CUDA_CHECK(cuCtxGetDevice(&device));
    auto known = fnaccRegistry.contexts.find(context);
    FNACCContextState &state = known != fnaccRegistry.contexts.end()
        ? known->second
        : fnaccCreateContextState(
              device, context, /*retainedPrimaryContext=*/false);
    fnaccRegistry.activeContext = state.context;
    return;
  }

  int ordinal = fnaccGetCudaDeviceOrdinal();
  FNACCContextState &state = fnaccGetOrCreatePrimaryContextState(ordinal);
  FNACC_CUDA_CHECK(cuCtxSetCurrent(state.context));
  fnaccRegistry.activeContext = state.context;

  if (fnaccDebugEnabled())
    std::fprintf(stderr,
        "FNACC: active CUDA device ordinal=%d device=%d context=%p\n", ordinal,
        static_cast<int>(state.device), static_cast<void *>(state.context));
}

static void fnaccWaitForStream(CUstream stream, CUevent completionEvent) {
  if (!stream || !completionEvent) {
    std::fprintf(stderr, "FNACC error: runtime stream is not initialized\n");
    std::abort();
  }
  FNACC_CUDA_CHECK(cuEventRecord(completionEvent, stream));
  FNACC_CUDA_CHECK(cuEventSynchronize(completionEvent));
}

static void fnaccWaitForRuntimeStream() {
  FNACCContextState &state = fnaccActiveContextState();
  fnaccWaitForStream(state.stream, state.completionEvent);
}

static void fnaccSynchronizeActiveContext() {
  if (!fnaccRegistry.activeContext)
    return;

  FNACCContextState &state{fnaccActiveContextState()};
  if (state.stream)
    fnaccWaitForStream(state.stream, state.completionEvent);
}

extern "C" void __fnacc_wait() {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: wait for active runtime stream\n");

  fnaccWaitForRuntimeStream();
}

static FNACCReductionWorkspace &fnaccGetReductionWorkspace() {
  fnaccEnsureCurrentContext();
  FNACCContextState &state = fnaccActiveContextState();
  FNACCReductionWorkspace &workspace = state.reductionWorkspace;
  if (!workspace.context) {
    workspace.device = state.device;
    workspace.context = state.context;
  }
  return workspace;
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

  auto &cache{fnaccActiveContextState().deviceCache};
  auto it{cache.find(hostPtr)};

  return it != cache.end() && it->second.ptr != 0 && it->second.bytes != 0;
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

static int32_t fnaccEffectiveWriteTargetForSlot(
    const FNACCKernelDesc *desc, int32_t slot, void *hostPtr) {
  if (desc && !desc->copyBackWrites)
    return FNACC_PACK_TARGET_DEVICE;
  return fnaccEffectivePackTargetForSlot(desc, slot, hostPtr);
}

static int32_t fnaccEffectivePackTargetForArray(
    const FNACCKernelDesc *desc, int32_t arrayIndex, void *hostPtr) {
  std::optional<int32_t> explicitTarget;
  if (desc) {
    for (const FNACCKernelParameterDesc &parameter : desc->parameters) {
      if (parameter.arrayIndex != arrayIndex ||
          (parameter.role != FNACCKernelParameterRole::Read &&
              parameter.role != FNACCKernelParameterRole::Write &&
              parameter.role != FNACCKernelParameterRole::ReadWrite))
        continue;
      std::optional<int32_t> candidate =
          fnaccExplicitPackTargetForSlot(desc, parameter.slot);
      if (!candidate)
        continue;
      if (explicitTarget && *explicitTarget != *candidate) {
        std::fprintf(stderr,
            "FNACC error: conflicting PACK targets for v2 array %d in "
            "kernel id %d\n",
            arrayIndex, desc->id);
        std::abort();
      }
      explicitTarget = candidate;
    }
  }
  if (explicitTarget)
    return *explicitTarget;
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

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);

  bool needAllocate = false;

  if (it == cache.end()) {
    needAllocate = true;
  } else if (it->second.bytes != bytes) {
    if (it->second.dataRegionReferences != 0) {
      std::fprintf(stderr,
          "FNACC error: cannot resize cached allocation for %s slot %d "
          "while it is owned by %zu data region(s)\n",
          role, slot, it->second.dataRegionReferences);
      std::abort();
    }
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: cache size mismatch for %s slot %d host=%p; "
          "old bytes=%zu new bytes=%zu, reallocating\n",
          role, slot, hostPtr, it->second.bytes, bytes);
    }

    fnaccSynchronizeActiveContext();

    FNACC_CUDA_CHECK(cuMemFree(it->second.ptr));
    cache.erase(it);
    needAllocate = true;
  }

  if (needAllocate) {
    FNACCDeviceAllocation allocation;
    allocation.bytes = bytes;
    FNACC_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnMiss)
      FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));

    auto inserted = cache.emplace(hostPtr, allocation);
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
  FNACCContextState &state = fnaccActiveContextState();

  auto cacheIt = state.functionCache.find(kernelId);
  if (cacheIt != state.functionCache.end())
    return cacheIt->second;

  std::string kernelName;

  if (const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId)) {
    kernelName = desc->name;
  } else {
    std::fprintf(
        stderr, "FNACC error: no JSON descriptor for kernel id %d\n", kernelId);
    std::abort();
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: looking up CUDA kernel id %d as symbol '%s'\n",
        kernelId, kernelName.c_str());
  }

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);

  std::size_t moduleIndex = 0;
  if (desc)
    moduleIndex = static_cast<std::size_t>(desc->ptxIndex);

  if (moduleIndex >= state.modules.size() || !state.modules[moduleIndex]) {
    std::fprintf(stderr,
        "FNACC error: kernel id %d requests device image index %zu, "
        "but only %zu module(s) are loaded\n",
        kernelId, moduleIndex, state.modules.size());
    std::abort();
  }

  CUfunction fn = nullptr;
  FNACC_CUDA_CHECK(
      cuModuleGetFunction(&fn, state.modules[moduleIndex], kernelName.c_str()));

  state.functionCache[kernelId] = fn;
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
        "FNACC error: host launch rank %d disagrees with JSON "
        "rank %d for kernel id %d\n",
        rank, desc->rank, kernelId);
    std::abort();
  }

  if (desc->tileX != blockX || desc->tileY != blockY || desc->tileZ != blockZ) {
    std::fprintf(stderr,
        "FNACC error: host tile (%d,%d,%d) disagrees with JSON "
        "tile (%d,%d,%d) for kernel id %d\n",
        blockX, blockY, blockZ, desc->tileX, desc->tileY, desc->tileZ,
        kernelId);
    std::abort();
  }
}

static unsigned fnaccCdiv(std::int64_t x, std::int64_t y, const char *what) {
  if (x < 0 || y <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid values while computing %s: "
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
        "FNACC error: grid dimension overflow while computing %s\n", what);
    std::abort();
  }

  return static_cast<unsigned>(result);
}

static std::size_t fnaccElementCount(
    int32_t rank, int32_t extentX, int32_t extentY, int32_t extentZ) {
  if (rank < 1 || rank > 3 || extentX < 0 || extentY < 0 || extentZ < 0) {
    std::fprintf(stderr, "FNACC error: invalid rank or negative extent\n");
    std::abort();
  }

  std::size_t count = static_cast<std::size_t>(extentX);

  if (rank >= 2)
    count = fnaccCheckedMul(
        count, static_cast<std::size_t>(extentY), "launch element count");

  if (rank >= 3)
    count = fnaccCheckedMul(
        count, static_cast<std::size_t>(extentZ), "launch element count");

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

extern "C" void __fnacc_validate_contiguous_desc(void *hostPtr,
    int64_t elementBytes, int32_t rank, int64_t extent0, int64_t extent1,
    int64_t extent2, int64_t stride0, int64_t stride1, int64_t stride2) {
  FNACC_RUNTIME_GUARD();
  if (!hostPtr && extent0 != 0 && extent1 != 0 && extent2 != 0) {
    std::fprintf(
        stderr, "FNACC error: launch descriptor has a null base pointer\n");
    std::abort();
  }
  fnaccValidateContiguousDescriptor("FNACC kernel launch", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);
}

extern "C" void __fnacc_validate_launch_desc(void *hostPtr,
    int64_t elementBytes, int32_t rank, int64_t lower0, int64_t lower1,
    int64_t lower2, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2, int32_t expectedRank,
    int64_t expectedExtent0, int64_t expectedExtent1, int64_t expectedExtent2) {
  FNACC_RUNTIME_GUARD();

  fnaccValidateContiguousDescriptor("FNACC kernel launch", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);

  if (rank != expectedRank) {
    std::fprintf(stderr,
        "FNACC error: descriptor rank %d does not match "
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
          "FNACC error: dimension %d has lower bound %lld; "
          "the current Triton lowering requires lower bound 1\n",
          dim + 1, static_cast<long long>(lowers[dim]));
      std::abort();
    }

    if (extents[dim] < expected[dim]) {
      std::fprintf(stderr,
          "FNACC error: dimension %d extent %lld is smaller "
          "than required launch extent %lld\n",
          dim + 1, static_cast<long long>(extents[dim]),
          static_cast<long long>(expected[dim]));
      std::abort();
    }
  }

  if (!hostPtr && expectedExtent0 > 0 && expectedExtent1 > 0 &&
      expectedExtent2 > 0) {
    std::fprintf(
        stderr, "FNACC error: nonempty launch has a null array pointer\n");
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
    count = fnaccCheckedMul(
        count, static_cast<std::size_t>(extent1), "descriptor element count");

  if (rank >= 3)
    count = fnaccCheckedMul(
        count, static_cast<std::size_t>(extent2), "descriptor element count");

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

  if (!hostPtr) {
    std::fprintf(stderr, "FNACC error: cannot cache a null host pointer\n");
    std::abort();
  }

  if (bytes == 0) {
    std::fprintf(stderr,
        "FNACC error: persistent FNACC allocations must have "
        "nonzero size\n");
    std::abort();
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);

  if (it != cache.end()) {
    if (it->second.bytes == bytes)
      return it->second;

    if (it->second.dataRegionReferences != 0) {
      std::fprintf(stderr,
          "FNACC error: %s cannot resize host=%p while it is owned by %zu "
          "data region(s)\n",
          operationName, hostPtr, it->second.dataRegionReferences);
      std::abort();
    }

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: %s resizing cached allocation for host=%p "
          "old_bytes=%zu new_bytes=%zu\n",
          operationName, hostPtr, it->second.bytes, bytes);
    }

    fnaccSynchronizeActiveContext();

    FNACC_CUDA_CHECK(cuMemFree(it->second.ptr));
    cache.erase(it);
  }

  FNACCDeviceAllocation allocation;
  allocation.bytes = bytes;

  if (bytes > 0) {
    FNACC_CUDA_CHECK(cuMemAlloc(&allocation.ptr, bytes));

    if (copyHostToDeviceOnCreateOrResize)
      FNACC_CUDA_CHECK(cuMemcpyHtoD(allocation.ptr, hostPtr, bytes));
  }

  auto inserted = cache.emplace(hostPtr, allocation);

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

static FNACCDataRegionFrame &fnaccCurrentDataRegion(const char *operationName) {
  auto &regions = fnaccActiveContextState().dataRegions;
  if (regions.empty()) {
    std::fprintf(stderr,
        "FNACC error: %s requires an active ENTER DATA region\n",
        operationName);
    std::abort();
  }
  return regions.back();
}

static bool fnaccFrameOwnsAllocation(
    const FNACCDataRegionFrame &frame, void *hostPtr) {
  return std::find(frame.allocations.begin(), frame.allocations.end(),
             hostPtr) != frame.allocations.end();
}

static FNACCDeviceAllocation &fnaccAcquireDataRegionAllocation(void *hostPtr,
    std::size_t bytes, bool copyHostToDeviceOnCreate,
    const char *operationName) {
  FNACCDataRegionFrame &frame = fnaccCurrentDataRegion(operationName);
  auto &cache = fnaccActiveContextState().deviceCache;

  if (fnaccFrameOwnsAllocation(frame, hostPtr)) {
    auto existing = cache.find(hostPtr);
    if (existing == cache.end()) {
      std::fprintf(stderr,
          "FNACC error: %s found corrupt data-region ownership for host=%p\n",
          operationName, hostPtr);
      std::abort();
    }
    if (existing->second.bytes != bytes) {
      std::fprintf(stderr,
          "FNACC error: %s repeats host=%p with a different size inside one "
          "data region (%zu versus %zu bytes)\n",
          operationName, hostPtr, existing->second.bytes, bytes);
      std::abort();
    }
    return existing->second;
  }

  FNACCDeviceAllocation *allocation = nullptr;
  if (bytes == 0) {
    auto inserted = cache.emplace(hostPtr, FNACCDeviceAllocation{});
    if (!inserted.second && inserted.first->second.bytes != 0) {
      std::fprintf(stderr,
          "FNACC error: %s cannot acquire zero-sized host=%p while a "
          "non-empty allocation is present\n",
          operationName, hostPtr);
      std::abort();
    }
    allocation = &inserted.first->second;
  } else {
    allocation = &fnaccGetOrCreateCachedAllocation(
        hostPtr, bytes, copyHostToDeviceOnCreate, operationName);
  }
  ++allocation->dataRegionReferences;
  frame.allocations.push_back(hostPtr);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s acquired host=%p in data region depth=%zu references=%zu\n",
        operationName, hostPtr, fnaccActiveContextState().dataRegions.size(),
        allocation->dataRegionReferences);
  }
  return *allocation;
}

static FNACCDeviceAllocation &fnaccAcquireExistingDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  FNACCDataRegionFrame &frame = fnaccCurrentDataRegion(operationName);
  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "FNACC error: %s has no sized cached allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }

  if (!fnaccFrameOwnsAllocation(frame, hostPtr)) {
    ++it->second.dataRegionReferences;
    frame.allocations.push_back(hostPtr);
  }
  return it->second;
}

static FNACCDeviceAllocation &fnaccGetOwnedDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  FNACCDataRegionFrame &frame = fnaccCurrentDataRegion(operationName);
  if (!fnaccFrameOwnsAllocation(frame, hostPtr)) {
    std::fprintf(stderr,
        "FNACC error: %s for host=%p does not belong to the innermost data "
        "region\n",
        operationName, hostPtr);
    std::abort();
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end() || it->second.dataRegionReferences == 0) {
    std::fprintf(stderr,
        "FNACC error: %s found corrupt data-region allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }
  return it->second;
}

static void fnaccCopyoutDataRegionAllocation(
    void *hostPtr, std::size_t bytes, const char *operationName) {
  FNACCDeviceAllocation &allocation =
      fnaccGetOwnedDataRegionAllocation(hostPtr, operationName);
  if (allocation.bytes < bytes) {
    std::fprintf(stderr,
        "FNACC error: %s requested %zu bytes for host=%p, but the cached "
        "allocation has only %zu bytes\n",
        operationName, bytes, hostPtr, allocation.bytes);
    std::abort();
  }

  // Present-or-copyout semantics: an inner region relinquishes only its own
  // reference. The host is updated only by the last owning region.
  if (allocation.dataRegionReferences == 1) {
    if (bytes != 0)
      FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, allocation.ptr, bytes));
  } else if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s deferred host copy for host=%p; %zu enclosing data "
        "region reference(s) remain\n",
        operationName, hostPtr, allocation.dataRegionReferences - 1);
  }
}

static void fnaccReleaseDataRegionAllocation(
    void *hostPtr, const char *operationName) {
  FNACCDataRegionFrame &frame = fnaccCurrentDataRegion(operationName);
  auto owned =
      std::find(frame.allocations.begin(), frame.allocations.end(), hostPtr);
  if (owned == frame.allocations.end()) {
    std::fprintf(stderr,
        "FNACC error: %s for host=%p does not belong to the innermost data "
        "region\n",
        operationName, hostPtr);
    std::abort();
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto allocation = cache.find(hostPtr);
  if (allocation == cache.end() ||
      allocation->second.dataRegionReferences == 0) {
    std::fprintf(stderr,
        "FNACC error: %s found corrupt data-region allocation for host=%p\n",
        operationName, hostPtr);
    std::abort();
  }

  frame.allocations.erase(owned);
  --allocation->second.dataRegionReferences;
  if (allocation->second.dataRegionReferences != 0)
    return;

  CUdeviceptr devicePtr = allocation->second.ptr;
  std::size_t bytes = allocation->second.bytes;
  fnaccSynchronizeActiveContext();
  if (devicePtr)
    FNACC_CUDA_CHECK(cuMemFree(devicePtr));
  cache.erase(allocation);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s released final data-region allocation host=%p "
        "device=0x%llx bytes=%zu\n",
        operationName, hostPtr, static_cast<unsigned long long>(devicePtr),
        bytes);
  }
}

extern "C" void __fnacc_enter_data_region() {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  fnaccActiveContextState().dataRegions.emplace_back();
  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: enter data region depth=%zu\n",
        fnaccActiveContextState().dataRegions.size());
}

extern "C" void __fnacc_exit_data_region() {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  auto &regions = fnaccActiveContextState().dataRegions;
  if (regions.empty()) {
    std::fprintf(stderr,
        "FNACC error: EXIT DATA has no matching active ENTER DATA region\n");
    std::abort();
  }

  while (!regions.back().allocations.empty()) {
    void *hostPtr = regions.back().allocations.back();
    fnaccReleaseDataRegionAllocation(hostPtr, "exit_data_region");
  }
  regions.pop_back();

  if (fnaccDebugEnabled())
    std::fprintf(stderr, "FNACC: exit data region depth=%zu\n", regions.size());
}

extern "C" void __fnacc_data_copyin_bytes(void *hostPtr, int64_t bytesValue) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: data_copyin_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  fnaccAcquireDataRegionAllocation(hostPtr,
      static_cast<std::size_t>(bytesValue),
      /*copyHostToDeviceOnCreate=*/true, "data_copyin_bytes");
}

extern "C" void __fnacc_data_create_bytes(void *hostPtr, int64_t bytesValue) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: data_create_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  fnaccAcquireDataRegionAllocation(hostPtr,
      static_cast<std::size_t>(bytesValue),
      /*copyHostToDeviceOnCreate=*/false, "data_create_bytes");
}

extern "C" void __fnacc_data_copyout_bytes(void *hostPtr, int64_t bytesValue) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: data_copyout_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  fnaccCopyoutDataRegionAllocation(
      hostPtr, static_cast<std::size_t>(bytesValue), "data_copyout_bytes");
}

extern "C" void __fnacc_data_copyin_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  fnaccValidateContiguousDescriptor("__fnacc_data_copyin_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  fnaccAcquireDataRegionAllocation(hostPtr, bytes,
      /*copyHostToDeviceOnCreate=*/true, "data_copyin_desc");
}

extern "C" void __fnacc_data_create_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  fnaccValidateContiguousDescriptor("__fnacc_data_create_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  fnaccAcquireDataRegionAllocation(hostPtr, bytes,
      /*copyHostToDeviceOnCreate=*/false, "data_create_desc");
}

extern "C" void __fnacc_data_copyout_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  fnaccValidateContiguousDescriptor("__fnacc_data_copyout_desc", elementBytes,
      rank, extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  fnaccCopyoutDataRegionAllocation(hostPtr, bytes, "data_copyout_desc");
}

extern "C" void __fnacc_data_copyin(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (hostPtr)
    fnaccAcquireExistingDataRegionAllocation(hostPtr, "data_copyin");
}

extern "C" void __fnacc_data_copyout(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;
  FNACCDeviceAllocation &allocation =
      fnaccGetOwnedDataRegionAllocation(hostPtr, "data_copyout");
  fnaccCopyoutDataRegionAllocation(hostPtr, allocation.bytes, "data_copyout");
}

extern "C" void __fnacc_data_delete(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  if (!hostPtr)
    return;

  // Validate that DELETE names an allocation owned by this frame. Actual
  // release is performed by the following data-region-exit marker so clause
  // source order cannot make DELETE run before COPYOUT.
  FNACCDeviceAllocation &allocation =
      fnaccGetOwnedDataRegionAllocation(hostPtr, "data_delete");
  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: data_delete marked host=%p device=0x%llx for release at "
        "region exit (references=%zu)\n",
        hostPtr, static_cast<unsigned long long>(allocation.ptr),
        allocation.dataRegionReferences);
  }
}

static void fnaccRequirePresentAllocation(
    const char *operationName, void *hostPtr, std::size_t requiredBytes) {
  if (!hostPtr) {
    std::fprintf(stderr, "FNACC error: %s received a null host pointer\n",
        operationName);
    std::abort();
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end() || it->second.ptr == 0 || it->second.bytes == 0) {
    std::fprintf(stderr,
        "FNACC error: %s requires host=%p to be present on the device; "
        "use enter data copyin/create first\n",
        operationName, hostPtr);
    std::abort();
  }

  if (requiredBytes != 0 && it->second.bytes < requiredBytes) {
    std::fprintf(stderr,
        "FNACC error: %s requires %zu bytes for host=%p, but the present "
        "allocation has only %zu bytes\n",
        operationName, requiredBytes, hostPtr, it->second.bytes);
    std::abort();
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s confirmed host=%p device=0x%llx bytes=%zu\n", operationName,
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        it->second.bytes);
  }
}

extern "C" void __fnacc_present(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();
  fnaccRequirePresentAllocation("present", hostPtr, 0);
}

extern "C" void __fnacc_present_bytes(void *hostPtr, int64_t bytesValue) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (bytesValue < 0) {
    std::fprintf(stderr,
        "FNACC error: present_bytes received negative byte count %lld\n",
        static_cast<long long>(bytesValue));
    std::abort();
  }
  if (bytesValue == 0)
    return;

  fnaccRequirePresentAllocation(
      "present_bytes", hostPtr, static_cast<std::size_t>(bytesValue));
}

extern "C" void __fnacc_present_desc(void *hostPtr, int64_t elementBytes,
    int32_t rank, int64_t extent0, int64_t extent1, int64_t extent2,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  fnaccValidateContiguousDescriptor("__fnacc_present_desc", elementBytes, rank,
      extent0, extent1, extent2, stride0, stride1, stride2);
  std::size_t bytes =
      fnaccBytesFromDescriptor(elementBytes, rank, extent0, extent1, extent2);
  if (bytes == 0)
    return;

  fnaccRequirePresentAllocation("present_desc", hostPtr, bytes);
}

extern "C" void __fnacc_create_bytes(void *hostPtr, int64_t bytesValue) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;

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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "FNACC error: update_host_bytes has no cached allocation for "
        "host=%p bytes=%zu; use create/copyin/update_device first\n",
        hostPtr, bytes);
    std::abort();
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "FNACC error: update_host_desc has no cached allocation for host=%p; "
        "use create/copyin/update_device first\n",
        hostPtr);
    std::abort();
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: release_desc ignored null pointer\n");
    return;
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release_desc ignored; no cached allocation for host=%p\n",
          hostPtr);
    }
    return;
  }

  if (it->second.dataRegionReferences != 0) {
    std::fprintf(stderr,
        "FNACC error: release_desc cannot release host=%p while it is owned "
        "by %zu data region(s); exit the owning region first\n",
        hostPtr, it->second.dataRegionReferences);
    std::abort();
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  fnaccSynchronizeActiveContext();

  FNACC_CUDA_CHECK(cuMemFree(devicePtr));
  cache.erase(it);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: release_desc host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __fnacc_launch_nd_f32(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, int32_t extentX, int32_t extentY, int32_t extentZ) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? fnaccCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned gridZ = 1;

  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes =
      fnaccCheckedMul(elemCount, sizeof(float), "f32 launch byte count");

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

    FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
        fnaccActiveContextState().stream, args, nullptr));
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

    FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
        fnaccActiveContextState().stream, args, nullptr));
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr, "FNACC: cuLaunchKernel rank2 returned\n");
      std::fflush(stderr);
    }
  }

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: waiting for runtime stream\n");
    std::fflush(stderr);
  }

  fnaccWaitForRuntimeStream();

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: runtime stream completed\n");
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
// Public runtime ABI v2: variadic metadata-driven elementwise launches
// -------------------------------------------------------------------------- //

struct FNACCPendingArrayV2 {
  void *host = nullptr;
  std::size_t bytes = 0;
  int32_t flags = 0;
  int64_t lower[3] = {1, 1, 1};
  int64_t stride[3] = {1, 0, 0};
  bool bound = false;
};

struct FNACCPendingScalarV2 {
  alignas(8) uint64_t storage = 0;
  std::size_t bytes = 0;
  bool bound = false;
};

struct FNACCPendingReductionResultV2 {
  void *host = nullptr;
  alignas(8) uint64_t initialStorage = 0;
  std::size_t bytes = 0;
  bool bound = false;
};

struct FNACCPendingLaunchV2 {
  bool active = false;
  CUcontext context = nullptr;
  int32_t kernelId = -1;
  int32_t rank = 0;
  int32_t block[3] = {1, 1, 1};
  int32_t extent[3] = {1, 1, 1};
  int32_t loopLower[3] = {1, 1, 1};
  std::vector<FNACCPendingArrayV2> arrays;
  std::vector<FNACCPendingScalarV2> scalars;
  std::vector<FNACCPendingReductionResultV2> reductionResults;
};

static thread_local FNACCPendingLaunchV2 fnaccPendingLaunchV2;

static void fnaccClearPendingLaunchV2() {
  fnaccPendingLaunchV2 = FNACCPendingLaunchV2{};
}

extern "C" void __fnacc_begin_launch_v2(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t extentX,
    int32_t extentY, int32_t extentZ, int32_t loopLowerX, int32_t loopLowerY,
    int32_t loopLowerZ, int32_t arrayCount, int32_t scalarCount) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (fnaccPendingLaunchV2.active) {
    std::fprintf(stderr,
        "FNACC error: nested or incomplete v2 launch on the same host "
        "thread\n");
    std::abort();
  }

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc || desc->launchAbiVersion != 2) {
    std::fprintf(stderr,
        "FNACC error: kernel id %d does not use launch ABI v2\n", kernelId);
    std::abort();
  }
  if (rank != desc->rank || rank < 1 || rank > 2 ||
      arrayCount != desc->arrayCount || scalarCount != desc->scalarCount ||
      arrayCount <= 0 || scalarCount < 0) {
    std::fprintf(stderr,
        "FNACC error: v2 launch ABI count/rank mismatch for kernel id %d\n",
        kernelId);
    std::abort();
  }
  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  FNACCPendingLaunchV2 &pending = fnaccPendingLaunchV2;
  pending.active = true;
  pending.context = fnaccRegistry.activeContext;
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

extern "C" void __fnacc_bind_array_v2(int32_t slot, void *host, int64_t bytes,
    int32_t flags, int64_t lowerX, int64_t lowerY, int64_t lowerZ,
    int64_t strideX, int64_t strideY, int64_t strideZ) {
  FNACC_RUNTIME_GUARD();
  if (!fnaccPendingLaunchV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >= fnaccPendingLaunchV2.arrays.size() ||
      !host || bytes <= 0 || (flags & 3) == 0 || (flags & ~3) != 0) {
    std::fprintf(stderr, "FNACC error: invalid v2 array binding\n");
    std::abort();
  }

  FNACCPendingArrayV2 &array = fnaccPendingLaunchV2.arrays[slot];
  if (array.bound) {
    std::fprintf(stderr, "FNACC error: v2 array slot %d bound twice\n", slot);
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

template <typename T> static void fnaccBindScalarV2(int32_t slot, T value) {
  FNACC_RUNTIME_GUARD();
  if (!fnaccPendingLaunchV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >= fnaccPendingLaunchV2.scalars.size()) {
    std::fprintf(stderr, "FNACC error: invalid v2 scalar binding\n");
    std::abort();
  }
  FNACCPendingScalarV2 &scalar = fnaccPendingLaunchV2.scalars[slot];
  if (scalar.bound) {
    std::fprintf(stderr, "FNACC error: v2 scalar slot %d bound twice\n", slot);
    std::abort();
  }
  static_assert(sizeof(T) <= sizeof(scalar.storage));
  std::memcpy(&scalar.storage, &value, sizeof(T));
  scalar.bytes = sizeof(T);
  scalar.bound = true;
}

#define FNACC_DEFINE_SCALAR_BINDER(SUFFIX, TYPE) \
  extern "C" void __fnacc_bind_scalar_##SUFFIX##_v2( \
      int32_t slot, TYPE value) { \
    fnaccBindScalarV2<TYPE>(slot, value); \
  }

FNACC_DEFINE_SCALAR_BINDER(i8, int8_t)
FNACC_DEFINE_SCALAR_BINDER(i16, int16_t)
FNACC_DEFINE_SCALAR_BINDER(i32, int32_t)
FNACC_DEFINE_SCALAR_BINDER(i64, int64_t)
FNACC_DEFINE_SCALAR_BINDER(f32, float)
FNACC_DEFINE_SCALAR_BINDER(f64, double)

#undef FNACC_DEFINE_SCALAR_BINDER

template <typename T>
static void fnaccBindReductionResultAtV2(
    int32_t slot, T *host, T initialValue) {
  FNACC_RUNTIME_GUARD();
  if (!fnaccPendingLaunchV2.active || slot < 0 ||
      static_cast<std::size_t>(slot) >=
          fnaccPendingLaunchV2.reductionResults.size() ||
      !host || fnaccPendingLaunchV2.reductionResults[slot].bound) {
    std::fprintf(stderr, "FNACC error: invalid v2 reduction result binding\n");
    std::abort();
  }
  FNACCPendingReductionResultV2 &result =
      fnaccPendingLaunchV2.reductionResults[slot];
  static_assert(sizeof(T) <= sizeof(result.initialStorage));
  result.host = host;
  std::memcpy(&result.initialStorage, &initialValue, sizeof(T));
  result.bytes = sizeof(T);
  result.bound = true;
}

#define FNACC_DEFINE_REDUCTION_RESULT_BINDER(SUFFIX, TYPE) \
  extern "C" void __fnacc_bind_reduction_result_##SUFFIX##_v2( \
      TYPE *host, TYPE initialValue) { \
    fnaccBindReductionResultAtV2<TYPE>(0, host, initialValue); \
  } \
  extern "C" void __fnacc_bind_reduction_result_##SUFFIX##_at_v2( \
      int32_t slot, TYPE *host, TYPE initialValue) { \
    fnaccBindReductionResultAtV2<TYPE>(slot, host, initialValue); \
  }

FNACC_DEFINE_REDUCTION_RESULT_BINDER(i8, int8_t)
FNACC_DEFINE_REDUCTION_RESULT_BINDER(i16, int16_t)
FNACC_DEFINE_REDUCTION_RESULT_BINDER(i32, int32_t)
FNACC_DEFINE_REDUCTION_RESULT_BINDER(i64, int64_t)
FNACC_DEFINE_REDUCTION_RESULT_BINDER(f32, float)
FNACC_DEFINE_REDUCTION_RESULT_BINDER(f64, double)

#undef FNACC_DEFINE_REDUCTION_RESULT_BINDER

extern "C" void __fnacc_launch_reduce_f32_v2(
    int32_t, int32_t, int32_t, float *, float *, float *, float, int32_t);
extern "C" void __fnacc_launch_reduce_f64_v2(
    int32_t, int32_t, int32_t, double *, double *, double *, double, int32_t);
extern "C" void __fnacc_launch_reduce_i8_v2(
    int32_t, int32_t, int32_t, int8_t *, int8_t *, int8_t *, int8_t, int32_t);
extern "C" void __fnacc_launch_reduce_i16_v2(int32_t, int32_t, int32_t,
    int16_t *, int16_t *, int16_t *, int16_t, int32_t);
extern "C" void __fnacc_launch_reduce_i32_v2(int32_t, int32_t, int32_t,
    int32_t *, int32_t *, int32_t *, int32_t, int32_t);
extern "C" void __fnacc_launch_reduce_i64_v2(int32_t, int32_t, int32_t,
    int64_t *, int64_t *, int64_t *, int64_t, int32_t);
extern "C" void __fnacc_launch_matmul_f32_v1(int32_t, int32_t, int32_t, int32_t,
    float *, float *, float *, int32_t, int32_t, int32_t);
extern "C" void __fnacc_launch_matmul_f64_v1(int32_t, int32_t, int32_t, int32_t,
    double *, double *, double *, int32_t, int32_t, int32_t);

static CUdeviceptr fnaccReserveReductionBuffer(FNACCDeviceAllocation &,
    std::size_t, FNACCReductionBufferStats &, const char *);
template <typename Real>
static Real fnaccReductionIdentity(FNACCKernelDesc::ReductionOperator);
template <typename Real>
static Real fnaccApplyReduction(FNACCKernelDesc::ReductionOperator, Real, Real);
template <typename Real>
static bool fnaccFinalizeReductionOnDevice(const FNACCKernelDesc *,
    FNACCReductionWorkspace &, CUdeviceptr, unsigned, Real *);
static int32_t fnaccCheckedI32Layout(int64_t, const char *);

static bool fnaccIsReductionKernelKind(const std::string &kind) {
  return kind == "reduction_sum1d" || kind == "reduction_dot1d" ||
      kind == "reduction_product1d" || kind == "reduction_min1d" ||
      kind == "reduction_max1d" || kind == "reduction_multi2d";
}

template <typename T>
static T fnaccPendingReductionInitial(
    const FNACCPendingReductionResultV2 &result) {
  T initialValue;
  std::memcpy(&initialValue, &result.initialStorage, sizeof(T));
  return initialValue;
}

template <typename T>
static void fnaccCommitReductionTypedV2(
    const FNACCKernelDesc *desc, FNACCPendingLaunchV2 &pending) {
  FNACCPendingReductionResultV2 &pendingResult = pending.reductionResults[0];
  T initialValue = fnaccPendingReductionInitial<T>(pendingResult);
  T *result = static_cast<T *>(pendingResult.host);
  if (pending.extent[0] <= 0) {
    *result = initialValue;
    fnaccClearPendingLaunchV2();
    return;
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned gridX = fnaccCdiv(
      pending.extent[0], pending.block[0], "v2 reduction grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(pending.kernelId);
  fnaccValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  fnaccValidateSupportedHiddenPtrArgCount(pending.kernelId);

  std::vector<FNACCDeviceArg> deviceArgs;
  std::vector<CUdeviceptr> devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    FNACCPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 1) == 0) {
      std::fprintf(stderr,
          "FNACC error: v2 reduction input is not readable for kernel id "
          "%d\n",
          pending.kernelId);
      std::abort();
    }
    int32_t target = fnaccEffectivePackTargetForArray(
        desc, static_cast<int32_t>(slot), array.host);
    FNACCDeviceArg device = fnaccPrepareReadBuffer(
        array.host, array.bytes, target, static_cast<int32_t>(slot));
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }

  FNACCReductionWorkspace &workspace = fnaccGetReductionWorkspace();
  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(T);
  CUdeviceptr dPartials = fnaccReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  std::vector<int32_t> parameterValues(desc->parameters.size(), 0);
  std::vector<void *> arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const FNACCKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case FNACCKernelParameterRole::Read:
      if ((pending.arrays[parameter.arrayIndex].flags & 1) == 0) {
        std::fprintf(stderr,
            "FNACC error: reduction array flags disagree with kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    case FNACCKernelParameterRole::Partials:
      arguments.push_back(&dPartials);
      break;
    case FNACCKernelParameterRole::Scalar: {
      FNACCPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != fnaccScalarParameterBytes(parameter.type)) {
        std::fprintf(stderr,
            "FNACC error: reduction scalar type mismatch for kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case FNACCKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case FNACCKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case FNACCKernelParameterRole::ArrayLowerBound:
    case FNACCKernelParameterRole::ArrayStride: {
      const FNACCPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = fnaccCheckedI32Layout(value,
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    default:
      std::fprintf(stderr,
          "FNACC error: unsupported reduction v2 parameter for kernel id %d "
          "slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  FNACCHiddenTritonArgs hidden;
  arguments.push_back(&hidden.hidden0);
  arguments.push_back(&hidden.hidden1);

  unsigned dynamicSharedBytes;
  if constexpr (std::is_same_v<T, double>) {
    dynamicSharedBytes =
        fnaccReductionF64DynamicSharedBytes(desc, pending.block[0]);
  } else if constexpr (std::is_integral_v<T>) {
    dynamicSharedBytes = fnaccReductionIntegerDynamicSharedBytes(
        desc, pending.block[0], sizeof(T));
  } else {
    dynamicSharedBytes =
        fnaccReductionDynamicSharedBytes(desc, pending.block[0]);
  }
  fnaccConfigureDynamicSharedMemory(
      function, pending.kernelId, dynamicSharedBytes);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch reduction v2 kernel id=%d arrays=%zu scalars=%zu "
        "grid=(%u,1,1) extent=%d\n",
        pending.kernelId, pending.arrays.size(), pending.scalars.size(), gridX,
        pending.extent[0]);
  }
  FNACC_CUDA_CHECK(cuLaunchKernel(function, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, arguments.data(),
      nullptr));
  ++workspace.primaryLaunches;

  T reducedValue = fnaccReductionIdentity<T>(desc->reductionOp);
  if (!fnaccFinalizeReductionOnDevice<T>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    fnaccWaitForRuntimeStream();
    std::vector<T> partials(gridX);
    FNACC_CUDA_CHECK(cuMemcpyDtoH(partials.data(), dPartials, partialBytes));
    for (T value : partials)
      reducedValue =
          fnaccApplyReduction(desc->reductionOp, reducedValue, value);
  }
  *result = fnaccApplyReduction(desc->reductionOp, initialValue, reducedValue);

  for (const FNACCDeviceArg &device : deviceArgs)
    fnaccReleaseDeviceArg(device);
  fnaccClearPendingLaunchV2();
}

template <typename T>
static void fnaccCommitMultiReductionTypedV2(
    const FNACCKernelDesc *desc, FNACCPendingLaunchV2 &pending) {
  if (pending.reductionResults.size() !=
          static_cast<std::size_t>(desc->outputCount) ||
      pending.reductionResults.empty()) {
    std::fprintf(stderr,
        "FNACC error: multi-reduction result count mismatch for kernel id "
        "%d\n",
        pending.kernelId);
    std::abort();
  }

  for (const FNACCPendingReductionResultV2 &result : pending.reductionResults)
    if (!result.bound || result.bytes != sizeof(T)) {
      std::fprintf(stderr,
          "FNACC error: incomplete multi-reduction result bindings for "
          "kernel id %d\n",
          pending.kernelId);
      std::abort();
    }

  if (pending.extent[0] <= 0 || pending.extent[1] <= 0) {
    for (FNACCPendingReductionResultV2 &result : pending.reductionResults)
      *static_cast<T *>(result.host) = fnaccPendingReductionInitial<T>(result);
    fnaccClearPendingLaunchV2();
    return;
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned gridX = fnaccCdiv(pending.extent[0], pending.block[0],
      "v2 multi-reduction grid dimension X");
  unsigned gridY = fnaccCdiv(pending.extent[1], pending.block[1],
      "v2 multi-reduction grid dimension Y");
  std::size_t programCountSize =
      fnaccCheckedMul(static_cast<std::size_t>(gridX),
          static_cast<std::size_t>(gridY), "multi-reduction program count");
  if (programCountSize >
      static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
    std::fprintf(stderr,
        "FNACC error: multi-reduction program count exceeds unsigned\n");
    std::abort();
  }
  unsigned programCount = static_cast<unsigned>(programCountSize);

  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(pending.kernelId);
  fnaccValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  fnaccValidateSupportedHiddenPtrArgCount(pending.kernelId);

  std::vector<FNACCDeviceArg> deviceArgs;
  std::vector<CUdeviceptr> devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    FNACCPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 1) == 0) {
      std::fprintf(stderr,
          "FNACC error: multi-reduction input is not readable for kernel id "
          "%d\n",
          pending.kernelId);
      std::abort();
    }
    int32_t target = fnaccEffectivePackTargetForArray(
        desc, static_cast<int32_t>(slot), array.host);
    FNACCDeviceArg device = fnaccPrepareReadBuffer(
        array.host, array.bytes, target, static_cast<int32_t>(slot));
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }

  FNACCReductionWorkspace &workspace = fnaccGetReductionWorkspace();
  std::size_t partialElements = fnaccCheckedMul(programCountSize,
      pending.reductionResults.size(), "multi-reduction partial elements");
  std::size_t partialBytes = fnaccCheckedMul(
      partialElements, sizeof(T), "multi-reduction partial bytes");
  CUdeviceptr dPartials = fnaccReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  std::vector<int32_t> parameterValues(desc->parameters.size(), 0);
  std::vector<void *> arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const FNACCKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case FNACCKernelParameterRole::Read:
      if ((pending.arrays[parameter.arrayIndex].flags & 1) == 0) {
        std::fprintf(stderr,
            "FNACC error: multi-reduction array flags disagree with kernel "
            "id %d slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    case FNACCKernelParameterRole::Partials:
      arguments.push_back(&dPartials);
      break;
    case FNACCKernelParameterRole::Scalar: {
      FNACCPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != fnaccScalarParameterBytes(parameter.type)) {
        std::fprintf(stderr,
            "FNACC error: multi-reduction scalar type mismatch for kernel "
            "id %d slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case FNACCKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case FNACCKernelParameterRole::ExtentY:
      arguments.push_back(&pending.extent[1]);
      break;
    case FNACCKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case FNACCKernelParameterRole::LoopLowerY:
      arguments.push_back(&pending.loopLower[1]);
      break;
    case FNACCKernelParameterRole::ArrayLowerBound:
    case FNACCKernelParameterRole::ArrayStride: {
      const FNACCPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = fnaccCheckedI32Layout(value,
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    default:
      std::fprintf(stderr,
          "FNACC error: unsupported multi-reduction v2 parameter for kernel "
          "id %d slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  FNACCHiddenTritonArgs hidden;
  arguments.push_back(&hidden.hidden0);
  arguments.push_back(&hidden.hidden1);

  std::size_t tileElementsSize =
      fnaccCheckedMul(static_cast<std::size_t>(pending.block[0]),
          static_cast<std::size_t>(pending.block[1]),
          "multi-reduction tile elements");
  if (tileElementsSize >
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    std::fprintf(stderr,
        "FNACC error: multi-reduction tile element count exceeds i32\n");
    std::abort();
  }
  int32_t tileElements = static_cast<int32_t>(tileElementsSize);
  unsigned dynamicSharedBytes;
  if constexpr (std::is_same_v<T, double>) {
    dynamicSharedBytes =
        fnaccReductionF64DynamicSharedBytes(desc, tileElements);
  } else if constexpr (std::is_integral_v<T>) {
    dynamicSharedBytes =
        fnaccReductionIntegerDynamicSharedBytes(desc, tileElements, sizeof(T));
  } else {
    dynamicSharedBytes = fnaccReductionDynamicSharedBytes(desc, tileElements);
  }
  fnaccConfigureDynamicSharedMemory(
      function, pending.kernelId, dynamicSharedBytes);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch multi-reduction v2 kernel id=%d arrays=%zu "
        "scalars=%zu outputs=%zu grid=(%u,%u,1) extent=(%d,%d)\n",
        pending.kernelId, pending.arrays.size(), pending.scalars.size(),
        pending.reductionResults.size(), gridX, gridY, pending.extent[0],
        pending.extent[1]);
  }
  FNACC_CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, arguments.data(),
      nullptr));
  ++workspace.primaryLaunches;

  std::size_t resultStrideBytes = fnaccCheckedMul(
      programCountSize, sizeof(T), "multi-reduction result partial stride");
  for (std::size_t index = 0; index < pending.reductionResults.size();
      ++index) {
    FNACCPendingReductionResultV2 &pendingResult =
        pending.reductionResults[index];
    T reducedValue = fnaccReductionIdentity<T>(desc->reductionOp);
    CUdeviceptr segment =
        dPartials + static_cast<CUdeviceptr>(index * resultStrideBytes);
    if (!fnaccFinalizeReductionOnDevice<T>(
            desc, workspace, segment, programCount, &reducedValue)) {
      fnaccWaitForRuntimeStream();
      std::vector<T> partials(programCount);
      FNACC_CUDA_CHECK(
          cuMemcpyDtoH(partials.data(), segment, resultStrideBytes));
      for (T value : partials)
        reducedValue =
            fnaccApplyReduction(desc->reductionOp, reducedValue, value);
    }
    T initialValue = fnaccPendingReductionInitial<T>(pendingResult);
    *static_cast<T *>(pendingResult.host) =
        fnaccApplyReduction(desc->reductionOp, initialValue, reducedValue);
  }

  for (const FNACCDeviceArg &device : deviceArgs)
    fnaccReleaseDeviceArg(device);
  fnaccClearPendingLaunchV2();
}

static bool fnaccTryCommitReductionLaunchV2(
    const FNACCKernelDesc *desc, FNACCPendingLaunchV2 &pending) {
  if (!fnaccIsReductionKernelKind(desc->kind))
    return false;
  if (pending.arrays.empty()) {
    std::fprintf(stderr,
        "FNACC error: incomplete v2 reduction bindings for kernel id %d\n",
        pending.kernelId);
    std::abort();
  }

  std::string pointerType;
  for (const FNACCKernelParameterDesc &parameter : desc->parameters)
    if (parameter.role == FNACCKernelParameterRole::Read) {
      pointerType = parameter.type;
      break;
    }
  bool isMultiReduction = desc->kind == "reduction_multi2d";
  if ((!isMultiReduction && pending.reductionResults.size() != 1) ||
      pending.reductionResults.empty()) {
    std::fprintf(stderr,
        "FNACC error: invalid v2 reduction result count for kernel id %d\n",
        pending.kernelId);
    std::abort();
  }
  std::size_t resultBytes = pending.reductionResults[0].bytes;
  for (const FNACCPendingReductionResultV2 &result : pending.reductionResults)
    if (!result.bound || result.bytes != resultBytes) {
      std::fprintf(stderr,
          "FNACC error: incomplete v2 reduction result bindings for kernel "
          "id %d\n",
          pending.kernelId);
      std::abort();
    }

#define FNACC_DISPATCH_REDUCTION(TYPE_NAME, TYPE) \
  if (pointerType == "ptr<" TYPE_NAME ">" && resultBytes == sizeof(TYPE)) { \
    if (isMultiReduction) \
      fnaccCommitMultiReductionTypedV2<TYPE>(desc, pending); \
    else \
      fnaccCommitReductionTypedV2<TYPE>(desc, pending); \
    return true; \
  }

  FNACC_DISPATCH_REDUCTION("i8", int8_t)
  FNACC_DISPATCH_REDUCTION("i16", int16_t)
  FNACC_DISPATCH_REDUCTION("i32", int32_t)
  FNACC_DISPATCH_REDUCTION("i64", int64_t)
  FNACC_DISPATCH_REDUCTION("f32", float)
  FNACC_DISPATCH_REDUCTION("f64", double)

#undef FNACC_DISPATCH_REDUCTION

  std::fprintf(stderr,
      "FNACC error: v2 reduction result type mismatch for kernel id %d\n",
      pending.kernelId);
  std::abort();
}

static bool fnaccTryCommitMatmulLaunchV2(
    const FNACCKernelDesc *desc, const FNACCPendingLaunchV2 &pending) {
  if (desc->kind != "matmul2d")
    return false;
  if (pending.arrays.size() != 3 || (pending.arrays[0].flags & 1) == 0 ||
      (pending.arrays[1].flags & 1) == 0 ||
      (pending.arrays[2].flags & 2) == 0) {
    std::fprintf(stderr,
        "FNACC error: matmul v2 requires readable A/B and writable C for "
        "kernel id %d\n",
        pending.kernelId);
    std::abort();
  }

  std::string pointerType;
  for (const FNACCKernelParameterDesc &parameter : desc->parameters)
    if (parameter.role == FNACCKernelParameterRole::Read) {
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
  fnaccClearPendingLaunchV2();

  if (pointerType == "ptr<f32>") {
    __fnacc_launch_matmul_f32_v1(kernelId, blockX, blockY, blockK,
        static_cast<float *>(a), static_cast<float *>(b),
        static_cast<float *>(c), n, m, k);
    return true;
  }
  if (pointerType == "ptr<f64>") {
    __fnacc_launch_matmul_f64_v1(kernelId, blockX, blockY, blockK,
        static_cast<double *>(a), static_cast<double *>(b),
        static_cast<double *>(c), n, m, k);
    return true;
  }

  std::fprintf(stderr,
      "FNACC error: unsupported matmul v2 element type for kernel id %d\n",
      kernelId);
  std::abort();
}

static int32_t fnaccCheckedI32Layout(int64_t value, const char *what) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    std::fprintf(
        stderr, "FNACC error: %s does not fit the device i32 ABI\n", what);
    std::abort();
  }
  return static_cast<int32_t>(value);
}

extern "C" void __fnacc_commit_launch_v2() {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  FNACCPendingLaunchV2 &pending = fnaccPendingLaunchV2;
  if (!pending.active || pending.context != fnaccRegistry.activeContext) {
    std::fprintf(stderr,
        "FNACC error: v2 launch committed without its original CUDA "
        "context\n");
    std::abort();
  }
  for (const FNACCPendingArrayV2 &array : pending.arrays)
    if (!array.bound) {
      std::fprintf(stderr, "FNACC error: incomplete v2 array bindings\n");
      std::abort();
    }
  for (const FNACCPendingScalarV2 &scalar : pending.scalars)
    if (!scalar.bound) {
      std::fprintf(stderr, "FNACC error: incomplete v2 scalar bindings\n");
      std::abort();
    }

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(pending.kernelId);
  bool isReduction = fnaccIsReductionKernelKind(desc->kind);
  bool anyReductionResultBound = std::any_of(pending.reductionResults.begin(),
      pending.reductionResults.end(),
      [](const FNACCPendingReductionResultV2 &result) { return result.bound; });
  bool allReductionResultsBound = !pending.reductionResults.empty() &&
      std::all_of(pending.reductionResults.begin(),
          pending.reductionResults.end(),
          [](const FNACCPendingReductionResultV2 &result) {
            return result.bound;
          });
  bool reductionResultsAgree = isReduction
      ? (pending.reductionResults.size() ==
                static_cast<std::size_t>(desc->outputCount) &&
            allReductionResultsBound)
      : !anyReductionResultBound;
  if (!reductionResultsAgree) {
    std::fprintf(stderr,
        "FNACC error: v2 reduction result binding disagrees with kernel id "
        "%d\n",
        pending.kernelId);
    std::abort();
  }
  if (fnaccTryCommitReductionLaunchV2(desc, pending))
    return;
  if (fnaccTryCommitMatmulLaunchV2(desc, pending))
    return;

  if (pending.extent[0] <= 0 || (pending.rank >= 2 && pending.extent[1] <= 0)) {
    fnaccClearPendingLaunchV2();
    return;
  }

  CUfunction function = getKernelFunction(pending.kernelId);
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(pending.kernelId);
  fnaccValidateCudaBlockSize(function, pending.kernelId, cudaBlockX);
  fnaccValidateSupportedHiddenPtrArgCount(pending.kernelId);

  std::vector<FNACCDeviceArg> deviceArgs;
  std::vector<CUdeviceptr> devicePointers;
  deviceArgs.reserve(pending.arrays.size());
  devicePointers.reserve(pending.arrays.size());
  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    FNACCPendingArrayV2 &array = pending.arrays[slot];
    int32_t target = fnaccEffectivePackTargetForArray(
        desc, static_cast<int32_t>(slot), array.host);
    if ((array.flags & 2) && !desc->copyBackWrites)
      target = FNACC_PACK_TARGET_DEVICE;
    FNACCDeviceArg device = (array.flags & 1)
        ? fnaccPrepareReadBuffer(
              array.host, array.bytes, target, static_cast<int32_t>(slot))
        : fnaccPrepareWriteBuffer(
              array.host, array.bytes, target, static_cast<int32_t>(slot));
    devicePointers.push_back(device.ptr);
    deviceArgs.push_back(device);
  }

  std::vector<int32_t> parameterValues(desc->parameters.size(), 0);
  std::vector<void *> arguments;
  arguments.reserve(desc->parameters.size() + 2);
  for (const FNACCKernelParameterDesc &parameter : desc->parameters) {
    switch (parameter.role) {
    case FNACCKernelParameterRole::Read:
    case FNACCKernelParameterRole::Write:
    case FNACCKernelParameterRole::ReadWrite: {
      FNACCPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int32_t requiredFlags = parameter.role == FNACCKernelParameterRole::Read
          ? 1
          : parameter.role == FNACCKernelParameterRole::Write ? 2
                                                              : 3;
      if ((array.flags & requiredFlags) != requiredFlags) {
        std::fprintf(stderr,
            "FNACC error: v2 array binding flags disagree with kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&devicePointers[parameter.arrayIndex]);
      break;
    }
    case FNACCKernelParameterRole::Scalar: {
      FNACCPendingScalarV2 &scalar = pending.scalars[parameter.scalarIndex];
      if (scalar.bytes != fnaccScalarParameterBytes(parameter.type)) {
        std::fprintf(stderr,
            "FNACC error: v2 scalar binding type mismatch for kernel id %d "
            "slot %d\n",
            pending.kernelId, parameter.slot);
        std::abort();
      }
      arguments.push_back(&scalar.storage);
      break;
    }
    case FNACCKernelParameterRole::ExtentX:
      arguments.push_back(&pending.extent[0]);
      break;
    case FNACCKernelParameterRole::ExtentY:
      arguments.push_back(&pending.extent[1]);
      break;
    case FNACCKernelParameterRole::ExtentZ:
      arguments.push_back(&pending.extent[2]);
      break;
    case FNACCKernelParameterRole::LoopLowerX:
      arguments.push_back(&pending.loopLower[0]);
      break;
    case FNACCKernelParameterRole::LoopLowerY:
      arguments.push_back(&pending.loopLower[1]);
      break;
    case FNACCKernelParameterRole::LoopLowerZ:
      arguments.push_back(&pending.loopLower[2]);
      break;
    case FNACCKernelParameterRole::ArrayLowerBound:
    case FNACCKernelParameterRole::ArrayStride: {
      const FNACCPendingArrayV2 &array = pending.arrays[parameter.arrayIndex];
      int64_t value =
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
          ? array.lower[parameter.dimension]
          : array.stride[parameter.dimension];
      parameterValues[parameter.slot] = fnaccCheckedI32Layout(value,
          parameter.role == FNACCKernelParameterRole::ArrayLowerBound
              ? "array lower bound"
              : "array stride");
      arguments.push_back(&parameterValues[parameter.slot]);
      break;
    }
    case FNACCKernelParameterRole::Partials:
    case FNACCKernelParameterRole::Unknown:
      std::fprintf(stderr,
          "FNACC error: unsupported parameter in v2 commit for kernel id %d "
          "slot %d\n",
          pending.kernelId, parameter.slot);
      std::abort();
    }
  }
  FNACCHiddenTritonArgs hidden;
  arguments.push_back(&hidden.hidden0);
  arguments.push_back(&hidden.hidden1);

  unsigned gridX =
      fnaccCdiv(pending.extent[0], pending.block[0], "v2 grid dimension X");
  unsigned gridY = pending.rank >= 2
      ? fnaccCdiv(pending.extent[1], pending.block[1], "v2 grid dimension Y")
      : 1;
  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: launch %s v2 kernel id=%d arrays=%zu scalars=%zu "
        "grid=(%u,%u,1) extent=(%d,%d) lower=(%d,%d)\n",
        desc->kind.c_str(), pending.kernelId, pending.arrays.size(),
        pending.scalars.size(), gridX, gridY, pending.extent[0],
        pending.extent[1], pending.loopLower[0], pending.loopLower[1]);
  }

  FNACC_CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, 1, cudaBlockX, 1, 1,
      0, fnaccActiveContextState().stream, arguments.data(), nullptr));
  fnaccWaitForRuntimeStream();

  for (std::size_t slot = 0; slot < pending.arrays.size(); ++slot) {
    FNACCPendingArrayV2 &array = pending.arrays[slot];
    if ((array.flags & 2) && deviceArgs[slot].target == FNACC_PACK_TARGET_HOST)
      fnaccCopyBackWriteBuffer(array.host, deviceArgs[slot], array.bytes);
  }
  for (const FNACCDeviceArg &device : deviceArgs)
    fnaccReleaseDeviceArg(device);

  fnaccClearPendingLaunchV2();
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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes =
      fnaccCheckedMul(numElems, sizeof(float), "f32 launch byte count");

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

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
      fnaccActiveContextState().stream, args, nullptr));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuLaunchKernel SAXPY returned\n");
    std::fflush(stderr);
  }

  fnaccWaitForRuntimeStream();

  FNACC_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  FNACC_CUDA_CHECK(cuMemFree(dA));
  FNACC_CUDA_CHECK(cuMemFree(dB));
  FNACC_CUDA_CHECK(cuMemFree(dC));
}

extern "C" void __fnacc_launch_nd_f32_s2(int32_t kernelId, int32_t rank,
    int32_t blockX, int32_t blockY, int32_t blockZ, float *a, float *b,
    float *c, float scalar0, float scalar1, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes =
      fnaccCheckedMul(numElems, sizeof(float), "f32 launch byte count");

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

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1, 0,
      fnaccActiveContextState().stream, args, nullptr));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: cuLaunchKernel s2 returned\n");
    std::fflush(stderr);
  }

  fnaccWaitForRuntimeStream();

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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? fnaccCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes =
      fnaccCheckedMul(elemCount, sizeof(float), "f32 launch byte count");

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

  int32_t writeTarget =
      fnaccEffectiveWriteTargetForSlot(desc, writeSlot, write);

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

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      fnaccActiveContextState().stream, args, nullptr));

  fnaccWaitForRuntimeStream();

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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? fnaccCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);
  std::size_t numBytes =
      fnaccCheckedMul(elemCount, sizeof(double), "f64 launch byte count");

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

  int32_t writeTarget =
      fnaccEffectiveWriteTargetForSlot(desc, writeSlot, write);

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

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      fnaccActiveContextState().stream, args, nullptr));

  fnaccWaitForRuntimeStream();

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

template <typename Integer>
static void fnaccLaunchIntegerV1(const char *abiName, const char *typeName,
    int32_t kernelId, int32_t rank, int32_t blockX, int32_t blockY,
    int32_t blockZ, int32_t numReadArrays, int32_t numScalars, Integer *read0,
    Integer *read1, Integer *read2, Integer *write, Integer scalar0,
    Integer scalar1, Integer scalar2, int32_t extentX, int32_t extentY,
    int32_t extentZ) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  fnaccValidateHostLaunchAgainstDesc(kernelId, rank, blockX, blockY, blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
        "FNACC error: %s requires one to three read arrays; "
        "got numReadArrays=%d for kernel id %d\n",
        abiName, numReadArrays, kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
        "FNACC error: unsupported numScalars=%d for kernel id %d\n", numScalars,
        kernelId);
    std::abort();
  }

  if (rank < 1 || rank > 3) {
    std::fprintf(
        stderr, "FNACC error: unsupported rank %d in %s\n", rank, abiName);
    std::abort();
  }

  if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
    std::fprintf(stderr,
        "FNACC error: invalid tile/block shape (%d,%d,%d) in %s\n", blockX,
        blockY, blockZ, abiName);
    std::abort();
  }

  if (!read0 || !write) {
    std::fprintf(stderr,
        "FNACC error: null required pointer in %s: read0=%p write=%p\n",
        abiName, static_cast<void *>(read0), static_cast<void *>(write));
    std::abort();
  }

  if (numReadArrays >= 2 && !read1) {
    std::fprintf(stderr, "FNACC error: null read1 pointer in %s\n", abiName);
    std::abort();
  }

  if (numReadArrays >= 3 && !read2) {
    std::fprintf(stderr, "FNACC error: null read2 pointer in %s\n", abiName);
    std::abort();
  }

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);
  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned gridY =
      rank >= 2 ? fnaccCdiv(extentY, blockY, "grid dimension Y") : 1;
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t elemCount = fnaccElementCount(rank, extentX, extentY, extentZ);
  std::size_t numBytes =
      fnaccCheckedMul(elemCount, sizeof(Integer), "integer launch byte count");

  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for generic %s kernel id %d\n",
        typeName, kernelId);
    std::abort();
  }

  if (desc->kind == "matmul2d") {
    std::fprintf(stderr,
        "FNACC error: generic %s launcher called for matmul kernel id %d\n",
        typeName, kernelId);
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
  int32_t writeTarget =
      fnaccEffectiveWriteTargetForSlot(desc, writeSlot, write);

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
        "FNACC: launch generic %s kernel id=%d rank=%d reads=%d scalars=%d "
        "grid=(%u,%u,1) tile=(%d,%d,%d) cuda_block=(%u,1,1) "
        "extent=(%d,%d,%d) bytes=%zu\n",
        typeName, kernelId, rank, numReadArrays, numScalars, gridX, gridY,
        blockX, blockY, blockZ, cudaBlockX, extentX, extentY, extentZ,
        numBytes);
  }

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);
  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, 1, cudaBlockX, 1, 1, 0,
      fnaccActiveContextState().stream, args, nullptr));
  fnaccWaitForRuntimeStream();

  if (writeDev.target == FNACC_PACK_TARGET_HOST) {
    fnaccCopyBackWriteBuffer(write, writeDev, numBytes);
  } else if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: skipped automatic copy-back for write slot %d "
        "because target=device; use !$fnacc update host(...) to copy back\n",
        writeDev.slot);
  }

  fnaccReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);
  if (numReadArrays >= 3)
    fnaccReleaseDeviceArg(read2Dev);
  fnaccReleaseDeviceArg(writeDev);
}

#define FNACC_DEFINE_INTEGER_LAUNCH(BITS, TYPE) \
  extern "C" void __fnacc_launch_i##BITS##_v1(int32_t kernelId, int32_t rank, \
      int32_t blockX, int32_t blockY, int32_t blockZ, int32_t numReadArrays, \
      int32_t numScalars, TYPE *read0, TYPE *read1, TYPE *read2, TYPE *write, \
      TYPE scalar0, TYPE scalar1, TYPE scalar2, int32_t extentX, \
      int32_t extentY, int32_t extentZ) { \
    fnaccLaunchIntegerV1<TYPE>("__fnacc_launch_i" #BITS "_v1", "i" #BITS, \
        kernelId, rank, blockX, blockY, blockZ, numReadArrays, numScalars, \
        read0, read1, read2, write, scalar0, scalar1, scalar2, extentX, \
        extentY, extentZ); \
  }

FNACC_DEFINE_INTEGER_LAUNCH(8, int8_t)
FNACC_DEFINE_INTEGER_LAUNCH(16, int16_t)
FNACC_DEFINE_INTEGER_LAUNCH(32, int32_t)
FNACC_DEFINE_INTEGER_LAUNCH(64, int64_t)

#undef FNACC_DEFINE_INTEGER_LAUNCH

extern "C" void __fnacc_launch_matmul_f32_v1(int32_t kernelId, int32_t blockX,
    int32_t blockY, int32_t blockK, float *a, float *b, float *c, int32_t n,
    int32_t m, int32_t k) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(n, blockX, "grid dimension X");
  unsigned gridY = fnaccCdiv(m, blockY, "grid dimension Y");
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
  int32_t cTarget = fnaccEffectiveWriteTargetForSlot(desc, 2, c);

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
      dynamicSharedBytes, fnaccActiveContextState().stream, args, nullptr));

  fnaccWaitForRuntimeStream();

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
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
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

  unsigned gridX = fnaccCdiv(n, blockX, "grid dimension X");
  unsigned gridY = fnaccCdiv(m, blockY, "grid dimension Y");
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
  int32_t cTarget = fnaccEffectiveWriteTargetForSlot(desc, 2, c);

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

  unsigned dynamicSharedBytes =
      fnaccMatmulF64DynamicSharedBytes(desc, blockX, blockY, blockK);

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
  fnaccConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, gridY, gridZ, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, args, nullptr));

  fnaccWaitForRuntimeStream();

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

static CUdeviceptr fnaccReserveReductionBuffer(
    FNACCDeviceAllocation &allocation, std::size_t requiredBytes,
    FNACCReductionBufferStats &stats, const char *bufferName) {
  if (requiredBytes == 0) {
    std::fprintf(stderr,
        "FNACC error: requested zero bytes for reduction %s buffer\n",
        bufferName);
    std::abort();
  }

  if (allocation.ptr && allocation.bytes >= requiredBytes) {
    ++stats.reuses;
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: reusing reduction %s buffer device=0x%llx "
          "capacity=%zu required=%zu\n",
          bufferName, static_cast<unsigned long long>(allocation.ptr),
          allocation.bytes, requiredBytes);
    }
    return allocation.ptr;
  }

  std::size_t oldBytes = allocation.bytes;
  if (allocation.ptr) {
    fnaccSynchronizeActiveContext();
    FNACC_CUDA_CHECK(cuMemFree(allocation.ptr));
  }

  allocation = {};
  FNACC_CUDA_CHECK(cuMemAlloc(&allocation.ptr, requiredBytes));
  allocation.bytes = requiredBytes;
  ++stats.allocations;
  if (oldBytes != 0)
    ++stats.growths;

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: %s reduction %s buffer device=0x%llx "
        "capacity=%zu\n",
        oldBytes == 0 ? "allocated" : "grew", bufferName,
        static_cast<unsigned long long>(allocation.ptr), allocation.bytes);
  }

  return allocation.ptr;
}

template <typename Real>
static Real fnaccReductionIdentity(FNACCKernelDesc::ReductionOperator op) {
  switch (op) {
  case FNACCKernelDesc::ReductionOperator::Add:
    return Real{0};
  case FNACCKernelDesc::ReductionOperator::Multiply:
    return Real{1};
  case FNACCKernelDesc::ReductionOperator::Min:
    if constexpr (std::numeric_limits<Real>::has_infinity)
      return std::numeric_limits<Real>::infinity();
    return std::numeric_limits<Real>::max();
  case FNACCKernelDesc::ReductionOperator::Max:
    if constexpr (std::numeric_limits<Real>::has_infinity)
      return -std::numeric_limits<Real>::infinity();
    return std::numeric_limits<Real>::lowest();
  }
  std::abort();
}

template <typename Real>
static Real fnaccApplyReduction(
    FNACCKernelDesc::ReductionOperator op, Real lhs, Real rhs) {
  switch (op) {
  case FNACCKernelDesc::ReductionOperator::Add:
    if constexpr (std::is_integral_v<Real>) {
      using Unsigned = std::make_unsigned_t<Real>;
      return static_cast<Real>(
          static_cast<Unsigned>(lhs) + static_cast<Unsigned>(rhs));
    }
    return lhs + rhs;
  case FNACCKernelDesc::ReductionOperator::Multiply:
    if constexpr (std::is_integral_v<Real>) {
      using Unsigned = std::make_unsigned_t<Real>;
      return static_cast<Real>(
          static_cast<Unsigned>(lhs) * static_cast<Unsigned>(rhs));
    }
    return lhs * rhs;
  case FNACCKernelDesc::ReductionOperator::Min:
    return rhs < lhs ? rhs : lhs;
  case FNACCKernelDesc::ReductionOperator::Max:
    return rhs > lhs ? rhs : lhs;
  }
  std::abort();
}

template <typename Real>
static bool fnaccFinalizeReductionOnDevice(const FNACCKernelDesc *primaryDesc,
    FNACCReductionWorkspace &workspace, CUdeviceptr dPartials,
    unsigned partialCount, Real *result) {
  if (!primaryDesc || primaryDesc->reductionStageId < 0)
    return false;

  int32_t stageKernelId = primaryDesc->reductionStageId;
  const FNACCKernelDesc *stageDesc = fnaccLookupKernelDesc(stageKernelId);

  if (!stageDesc) {
    std::fprintf(stderr,
        "FNACC error: reduction kernel id %d references missing stage "
        "kernel id %d\n",
        primaryDesc->id, stageKernelId);
    std::abort();
  }

  if (stageDesc->kind != "reduction_stage1d") {
    std::fprintf(stderr,
        "FNACC error: reduction stage kernel id %d has unexpected kind "
        "'%s'\n",
        stageKernelId, stageDesc->kind.c_str());
    std::abort();
  }

  if (stageDesc->reductionOp != primaryDesc->reductionOp) {
    std::fprintf(stderr,
        "FNACC error: primary reduction kernel id %d and stage kernel id %d "
        "use different reduction operators\n",
        primaryDesc->id, stageKernelId);
    std::abort();
  }

  int32_t stageBlock = stageDesc->tileX;
  if (stageBlock <= 1) {
    std::fprintf(stderr,
        "FNACC error: reduction stage kernel id %d requires tile_x > 1, "
        "got %d\n",
        stageKernelId, stageBlock);
    std::abort();
  }

  CUfunction stageFn = getKernelFunction(stageKernelId);
  unsigned stageCudaBlockX = fnaccCudaThreadsPerCTA(stageKernelId);

  fnaccValidateSupportedHiddenPtrArgCount(stageKernelId);
  fnaccValidateCudaBlockSize(stageFn, stageKernelId, stageCudaBlockX);

  CUdeviceptr current = dPartials;
  CUdeviceptr next = 0;

  if (partialCount > 1) {
    unsigned scratchElements = fnaccCdiv(
        static_cast<int32_t>(partialCount), stageBlock, "scratch Elements");
    std::size_t scratchBytes =
        fnaccCheckedMul(static_cast<std::size_t>(scratchElements), sizeof(Real),
            "hierarchical reduction scratch buffer");
    next = fnaccReserveReductionBuffer(
        workspace.scratch, scratchBytes, workspace.scratchStats, "scratch");
  }

  while (partialCount > 1) {
    if (partialCount >
        static_cast<unsigned>(std::numeric_limits<int32_t>::max())) {
      std::fprintf(stderr, "FNACC error: reduction stage extent exceeds i32\n");
      std::abort();
    }
    int32_t stageExtent = static_cast<int32_t>(partialCount);
    unsigned outputCount = fnaccCdiv(stageExtent, stageBlock, "output Count");
    FNACCHiddenTritonArgs hidden;

    void *args[] = {
        &current,
        &next,
        &stageExtent,
        &hidden.hidden0,
        &hidden.hidden1,
    };

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: launch reduction stage kernel id=%d input_count=%u "
          "output_count=%u tile=%d cuda_block=%u\n",
          stageKernelId, partialCount, outputCount, stageBlock,
          stageCudaBlockX);
    }

    unsigned dynamicSharedBytes =
        fnaccReductionDynamicSharedBytes(stageDesc, stageBlock);

    fnaccConfigureDynamicSharedMemory(
        stageFn, stageKernelId, dynamicSharedBytes);

    FNACC_CUDA_CHECK(cuLaunchKernel(stageFn, outputCount, 1, 1, stageCudaBlockX,
        1, 1, dynamicSharedBytes, fnaccActiveContextState().stream, args,
        nullptr));
    ++workspace.stageLaunches;

    CUdeviceptr oldCurrent = current;
    current = next;
    next = oldCurrent;
    partialCount = outputCount;
  }

  fnaccWaitForRuntimeStream();
  FNACC_CUDA_CHECK(cuMemcpyDtoH(result, current, sizeof(Real)));

  return true;
}

extern "C" void __fnacc_launch_reduce_f32_v2(int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, float *read0, float *read1, float *result,
    float initialValue, int32_t extentX) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(
        stderr, "FNACC error: null pointer in __fnacc_launch_reduce_f32_v2\n");
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "FNACC error: reduction f32 supports one or two read arrays, got %d\n",
        numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  FNACCReductionWorkspace &workspace = fnaccGetReductionWorkspace();

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(float);

  int32_t read0Target = fnaccEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? fnaccEffectivePackTargetForSlot(desc, 1, read1)
      : FNACC_PACK_TARGET_HOST;

  FNACCDeviceArg read0Dev =
      fnaccPrepareReadBuffer(read0, bytes, read0Target, 0);

  FNACCDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = fnaccPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;

  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(float);
  CUdeviceptr dPartials = fnaccReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

  void *args[8];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes = fnaccReductionDynamicSharedBytes(desc, blockX);

  fnaccConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  float reducedValue = fnaccReductionIdentity<float>(desc->reductionOp);

  if (!fnaccFinalizeReductionOnDevice<float>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    fnaccWaitForRuntimeStream();

    std::vector<float> partials(gridX);
    FNACC_CUDA_CHECK(cuMemcpyDtoH(partials.data(), dPartials, partialBytes));

    for (float value : partials)
      reducedValue =
          fnaccApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = fnaccApplyReduction(desc->reductionOp, initialValue, reducedValue);

  fnaccReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);
}

extern "C" void __fnacc_launch_reduce_f64_v2(int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, double *read0, double *read1, double *result,
    double initialValue, int32_t extentX) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(
        stderr, "FNACC error: null pointer in __fnacc_launch_reduce_f64_v2\n");
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "FNACC error: reduction f64 supports one or two read arrays, got %d\n",
        numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  FNACCReductionWorkspace &workspace = fnaccGetReductionWorkspace();

  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);

  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(double);

  int32_t read0Target = fnaccEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? fnaccEffectivePackTargetForSlot(desc, 1, read1)
      : FNACC_PACK_TARGET_HOST;

  FNACCDeviceArg read0Dev =
      fnaccPrepareReadBuffer(read0, bytes, read0Target, 0);

  FNACCDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = fnaccPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;

  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(double);
  CUdeviceptr dPartials = fnaccReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;

  void *args[8];
  int argCount = 0;

  args[argCount++] = &dRead0;

  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;

  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes =
      fnaccReductionF64DynamicSharedBytes(desc, blockX);

  fnaccConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  double reducedValue = fnaccReductionIdentity<double>(desc->reductionOp);

  if (!fnaccFinalizeReductionOnDevice<double>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    fnaccWaitForRuntimeStream();

    std::vector<double> partials(gridX);
    FNACC_CUDA_CHECK(cuMemcpyDtoH(partials.data(), dPartials, partialBytes));

    for (double value : partials)
      reducedValue =
          fnaccApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = fnaccApplyReduction(desc->reductionOp, initialValue, reducedValue);

  fnaccReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);
}

template <typename Integer>
static void fnaccLaunchReduceIntegerV2(const char *abiName,
    const char *typeName, int32_t kernelId, int32_t blockX,
    int32_t numReadArrays, Integer *read0, Integer *read1, Integer *result,
    Integer initialValue, int32_t extentX) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!read0 || !result) {
    std::fprintf(stderr, "FNACC error: null pointer in %s\n", abiName);
    std::abort();
  }

  if (numReadArrays < 1 || numReadArrays > 2) {
    std::fprintf(stderr,
        "FNACC error: reduction %s supports one or two read arrays, got %d\n",
        typeName, numReadArrays);
    std::abort();
  }

  CUfunction fn = getKernelFunction(kernelId);
  const FNACCKernelDesc *desc = fnaccLookupKernelDesc(kernelId);
  if (!desc) {
    std::fprintf(stderr,
        "FNACC error: no JSON descriptor for reduction kernel id %d\n",
        kernelId);
    std::abort();
  }

  if (extentX <= 0) {
    *result = initialValue;
    return;
  }

  FNACCReductionWorkspace &workspace = fnaccGetReductionWorkspace();
  unsigned gridX = fnaccCdiv(extentX, blockX, "grid dimension X");
  unsigned cudaBlockX = fnaccCudaThreadsPerCTA(kernelId);
  std::size_t bytes = static_cast<std::size_t>(extentX) * sizeof(Integer);

  int32_t read0Target = fnaccEffectivePackTargetForSlot(desc, 0, read0);
  int32_t read1Target = numReadArrays >= 2
      ? fnaccEffectivePackTargetForSlot(desc, 1, read1)
      : FNACC_PACK_TARGET_HOST;

  FNACCDeviceArg read0Dev =
      fnaccPrepareReadBuffer(read0, bytes, read0Target, 0);
  FNACCDeviceArg read1Dev;
  if (numReadArrays >= 2)
    read1Dev = fnaccPrepareReadBuffer(read1, bytes, read1Target, 1);

  CUdeviceptr dRead0 = read0Dev.ptr;
  CUdeviceptr dRead1 = read1Dev.ptr;
  std::size_t partialBytes = static_cast<std::size_t>(gridX) * sizeof(Integer);
  CUdeviceptr dPartials = fnaccReserveReductionBuffer(
      workspace.partials, partialBytes, workspace.partialStats, "partials");

  fnaccValidateSupportedHiddenPtrArgCount(kernelId);
  FNACCHiddenTritonArgs hidden;
  void *args[8];
  int argCount = 0;
  args[argCount++] = &dRead0;
  if (numReadArrays >= 2)
    args[argCount++] = &dRead1;
  args[argCount++] = &dPartials;
  args[argCount++] = &extentX;
  args[argCount++] = &hidden.hidden0;
  args[argCount++] = &hidden.hidden1;

  fnaccValidateCudaBlockSize(fn, kernelId, cudaBlockX);

  unsigned dynamicSharedBytes =
      fnaccReductionIntegerDynamicSharedBytes(desc, blockX, sizeof(Integer));

  fnaccConfigureDynamicSharedMemory(fn, kernelId, dynamicSharedBytes);

  FNACC_CUDA_CHECK(cuLaunchKernel(fn, gridX, 1, 1, cudaBlockX, 1, 1,
      dynamicSharedBytes, fnaccActiveContextState().stream, args, nullptr));
  ++workspace.primaryLaunches;

  Integer reducedValue = fnaccReductionIdentity<Integer>(desc->reductionOp);
  if (!fnaccFinalizeReductionOnDevice<Integer>(
          desc, workspace, dPartials, gridX, &reducedValue)) {
    fnaccWaitForRuntimeStream();

    std::vector<Integer> partials(gridX);
    FNACC_CUDA_CHECK(cuMemcpyDtoH(partials.data(), dPartials, partialBytes));
    for (Integer value : partials)
      reducedValue =
          fnaccApplyReduction(desc->reductionOp, reducedValue, value);
  }

  *result = fnaccApplyReduction(desc->reductionOp, initialValue, reducedValue);

  fnaccReleaseDeviceArg(read0Dev);
  if (numReadArrays >= 2)
    fnaccReleaseDeviceArg(read1Dev);
}

#define FNACC_DEFINE_INTEGER_REDUCTION(BITS, TYPE) \
  extern "C" void __fnacc_launch_reduce_i##BITS##_v2(int32_t kernelId, \
      int32_t blockX, int32_t numReadArrays, TYPE *read0, TYPE *read1, \
      TYPE *result, TYPE initialValue, int32_t extentX) { \
    fnaccLaunchReduceIntegerV2<TYPE>("__fnacc_launch_reduce_i" #BITS "_v2", \
        "i" #BITS, kernelId, blockX, numReadArrays, read0, read1, result, \
        initialValue, extentX); \
  }

FNACC_DEFINE_INTEGER_REDUCTION(8, int8_t)
FNACC_DEFINE_INTEGER_REDUCTION(16, int16_t)
FNACC_DEFINE_INTEGER_REDUCTION(32, int32_t)
FNACC_DEFINE_INTEGER_REDUCTION(64, int64_t)

#undef FNACC_DEFINE_INTEGER_REDUCTION

extern "C" void __fnacc_get_reduction_workspace_stats_v1(
    uint64_t *primaryLaunches, uint64_t *stageLaunches,
    uint64_t *partialAllocations, uint64_t *partialGrowths,
    uint64_t *partialReuses, uint64_t *partialCapacityBytes,
    uint64_t *scratchAllocations, uint64_t *scratchGrowths,
    uint64_t *scratchReuses, uint64_t *scratchCapacityBytes) {
  FNACC_RUNTIME_GUARD();
  FNACCReductionWorkspace workspace = fnaccAggregateReductionWorkspaceStats();
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
extern "C" void __fnacc_update_host(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_host ignored null pointer\n");
    return;
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "FNACC error: update_host has no cached allocation for %p; "
        "use create/copyin/update_device first\n",
        hostPtr);
    std::abort();
  }

  FNACC_CUDA_CHECK(cuMemcpyDtoH(hostPtr, it->second.ptr, it->second.bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: update_host host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(it->second.ptr),
        it->second.bytes);
  }
}

extern "C" void __fnacc_update_device(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: update_device ignored null pointer\n");
    return;
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    std::fprintf(stderr,
        "FNACC error: update_device has no cached allocation for %p; "
        "use a sized update/create directive first\n",
        hostPtr);
    std::abort();
  }

  FNACC_CUDA_CHECK(cuMemcpyHtoD(it->second.ptr, hostPtr, it->second.bytes));

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: update_device host=%p device=0x%llx bytes=%zu\n", hostPtr,
        static_cast<unsigned long long>(it->second.ptr), it->second.bytes);
  }
}

extern "C" void __fnacc_release(void *hostPtr) {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  if (!hostPtr) {
    if (fnaccDebugEnabled())
      std::fprintf(stderr, "FNACC: release ignored null pointer\n");
    return;
  }

  auto &cache = fnaccActiveContextState().deviceCache;
  auto it = cache.find(hostPtr);
  if (it == cache.end()) {
    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release ignored; no cached allocation for %p\n", hostPtr);
    }
    return;
  }

  if (it->second.dataRegionReferences != 0) {
    std::fprintf(stderr,
        "FNACC error: release cannot release host=%p while it is owned by "
        "%zu data region(s); exit the owning region first\n",
        hostPtr, it->second.dataRegionReferences);
    std::abort();
  }

  CUdeviceptr devicePtr = it->second.ptr;
  std::size_t bytes = it->second.bytes;

  fnaccSynchronizeActiveContext();

  FNACC_CUDA_CHECK(cuMemFree(devicePtr));

  cache.erase(it);

  if (fnaccDebugEnabled()) {
    std::fprintf(stderr, "FNACC: release host=%p device=0x%llx bytes=%zu\n",
        hostPtr, static_cast<unsigned long long>(devicePtr), bytes);
  }
}

extern "C" void __fnacc_release_all() {
  FNACC_RUNTIME_GUARD();
  FNACCCurrentContextGuard contextGuard;
  fnaccEnsureCurrentContext();

  auto &cache = fnaccActiveContextState().deviceCache;
  auto &regions = fnaccActiveContextState().dataRegions;
  if (!regions.empty()) {
    std::fprintf(stderr,
        "FNACC error: release_all cannot be used while %zu data region(s) "
        "are active; exit the regions first\n",
        regions.size());
    std::abort();
  }
  if (fnaccDebugEnabled()) {
    std::fprintf(stderr,
        "FNACC: release_all releasing %zu cached allocations\n", cache.size());
  }

  for (auto &entry : cache) {
    void *hostPtr = entry.first;
    FNACCDeviceAllocation &allocation = entry.second;

    if (fnaccDebugEnabled()) {
      std::fprintf(stderr,
          "FNACC: release_all host=%p device=0x%llx bytes=%zu\n", hostPtr,
          static_cast<unsigned long long>(allocation.ptr), allocation.bytes);
    }
    fnaccSynchronizeActiveContext();
    if (allocation.ptr)
      FNACC_CUDA_CHECK(cuMemFree(allocation.ptr));
  }

  cache.clear();
}

static void fnaccRegisterEmbeddedDeviceBundle(const void *const *imageData,
    const std::size_t *imageSizes, const int32_t *imageKinds,
    std::size_t imageCount, const char *jsonData, std::size_t jsonSize) {
  if (!imageData || !imageSizes || !imageKinds || imageCount == 0 ||
      !jsonData || jsonSize == 0) {
    std::fprintf(stderr, "FNACC error: invalid embedded kernel bundle\n");
    std::abort();
  }
  for (std::size_t i = 0; i < imageCount; ++i) {
    if (!imageData[i] || imageSizes[i] == 0 ||
        (imageKinds[i] != FNACCEmbeddedKernelBundle::PTX &&
            imageKinds[i] != FNACCEmbeddedKernelBundle::Cubin)) {
      std::fprintf(
          stderr, "FNACC error: invalid embedded device image entry %zu\n", i);
      std::abort();
    }
  }

  FNACCEmbeddedKernelBundle bundle;
  bundle.imageData.assign(imageData, imageData + imageCount);
  bundle.imageSize.assign(imageSizes, imageSizes + imageCount);
  bundle.imageKind.assign(imageKinds, imageKinds + imageCount);
  bundle.jsonData = jsonData;
  bundle.jsonSize = jsonSize;
  fnaccGetEmbeddedKernelBundles().push_back(std::move(bundle));
}

extern "C" void __fnacc_register_embedded_device_bundle(
    const void *const *imageData, const std::size_t *imageSizes,
    const int32_t *imageKinds, std::size_t imageCount, const char *jsonData,
    std::size_t jsonSize) {
  FNACC_RUNTIME_GUARD();
  fnaccRegisterEmbeddedDeviceBundle(
      imageData, imageSizes, imageKinds, imageCount, jsonData, jsonSize);
}

extern "C" void __fnacc_register_embedded_kernel_bundle(
    const char *const *ptxData, std::size_t const *ptxSizes,
    std::size_t ptxCount, const char *jsonData, std::size_t jsonSize) {
  FNACC_RUNTIME_GUARD();
  std::vector<const void *> imageData(ptxCount);
  for (std::size_t i = 0; i < ptxCount; ++i)
    imageData[i] = ptxData ? ptxData[i] : nullptr;
  std::vector<int32_t> imageKinds(ptxCount, FNACCEmbeddedKernelBundle::PTX);
  fnaccRegisterEmbeddedDeviceBundle(imageData.data(), ptxSizes,
      imageKinds.data(), ptxCount, jsonData, jsonSize);
}

extern "C" void __fnacc_register_embedded_kernels(const char *ptxData,
    std::size_t ptxSize, const char *jsonData, std::size_t jsonSize) {
  FNACC_RUNTIME_GUARD();
  const void *imageData[] = {ptxData};
  std::size_t imageSizes[] = {ptxSize};
  int32_t imageKinds[] = {FNACCEmbeddedKernelBundle::PTX};
  fnaccRegisterEmbeddedDeviceBundle(
      imageData, imageSizes, imageKinds, 1, jsonData, jsonSize);
}

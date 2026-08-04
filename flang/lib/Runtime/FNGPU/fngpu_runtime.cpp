#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

  // -------------------------------------------------------------------------- //
  // Tiny dependency-free JSON helpers
  // -------------------------------------------------------------------------- //
  //
  // These intentionally parse only the JSON shape emitted by the FNGPU compiler.
  // This is not a general-purpose JSON parser.
  //
  // Expected generated object form:
  //
  // {
  //   "id": 5,
  //   "name": "fngpu_kernel_5",
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

  static bool jsonFindInt(const std::string &text,
			  const char *key,
			  int32_t &out) {
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

  static bool jsonFindString(const std::string &text,
			     const char *key,
			     std::string &out) {
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

  static const char *skipToNextIntegerLike(const char *cursor,
					   const char *endOfString) {
    while (cursor < endOfString) {
      char c = *cursor;
      if ((c >= '0' && c <= '9') || c == '-')
	return cursor;
      ++cursor;
    }

    return cursor;
  }

  static bool jsonFindIntArray3(const std::string &text,
				const char *key,
				int32_t &x,
				int32_t &y,
				int32_t &z) {
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

  static std::size_t findKernelObjectEnd(const std::string &json,
					 std::size_t objectStart) {
    std::size_t nextKernelId = json.find("\n      \"id\"", objectStart + 1);
    if (nextKernelId != std::string::npos)
      return nextKernelId;

    std::size_t kernelsEnd = json.find("\n  ]", objectStart);
    if (kernelsEnd != std::string::npos)
      return kernelsEnd;

    return json.size();
  }
  
  [[noreturn]] static void fngpuFatal(const char *message) {
    std::fprintf(stderr, "FNGPU error: %s\n", message);
    std::abort();
  }

  static void fngpuCudaCheck(CUresult result,
			     const char *expr,
			     const char *file,
			     int line) {
    if (result == CUDA_SUCCESS)
      return;

    const char *name = nullptr;
    const char *desc = nullptr;

    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);

    std::fprintf(stderr,
		 "FNGPU CUDA driver error at %s:%d while executing %s: %s: %s\n",
		 file,
		 line,
		 expr,
		 name ? name : "<unknown>",
		 desc ? desc : "<no description>");

    std::abort();
  }

#define FNGPU_CUDA_CHECK(expr)				\
  do {							\
    fngpuCudaCheck((expr), #expr, __FILE__, __LINE__);	\
  } while (false)

  struct FNGPUHiddenTritonArgs {
    // Triton/NVVM-generated PTX currently appends two hidden pointer parameters
    // after the explicit kernel parameters. For the kernels FNGPU currently
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

  struct FNGPUKernelDesc {
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

    // Number of hidden pointer parameters appended by the Triton/NVVM generated
    // PTX after the explicit kernel arguments.
    //
    // Current generated kernels use two hidden pointer args.
    int32_t tritonHiddenPtrArgs = 2;
  };

  struct FNGPUKernelRegistry {
    bool initialized = false;

    CUcontext context = nullptr;
    CUmodule module = nullptr;

    std::unordered_map<int32_t, FNGPUKernelDesc> kernels;
    std::unordered_map<int32_t, CUfunction> functionCache;
  };

  static FNGPUKernelRegistry fngpuRegistry;

  static bool fngpuDebugEnabled() {
    const char *value = std::getenv("FNGPU_DEBUG");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }

  static const char *fngpuGetEnvOrDefault(const char *envName,
					  const char *fallback) {
    const char *value = std::getenv(envName);
    if (value && value[0] != '\0')
      return value;

    return fallback;
  }

  static std::string fngpuReadTextFile(const char *path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
      std::fprintf(stderr, "FNGPU error: could not open file '%s'\n", path);
      std::abort();
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
  }

  static std::unordered_map<int32_t, FNGPUKernelDesc> fngpuParseKernelDescsFromJson(const std::string &json) {
    std::unordered_map<int32_t, FNGPUKernelDesc> result;

    std::size_t pos = 0;

    while (true) {
      std::size_t idKey = json.find("\"id\"", pos);
      if (idKey == std::string::npos)
	break;

      std::size_t objectEnd = findKernelObjectEnd(json, idKey);
      std::string objectText = json.substr(idKey, objectEnd - idKey);

      FNGPUKernelDesc desc;

      if (!jsonFindInt(objectText, "id", desc.id)) {
	pos = objectEnd;
	continue;
      }

      if (!jsonFindString(objectText, "name", desc.name))
	desc.name = "fngpu_kernel_" + std::to_string(desc.id);

      jsonFindString(objectText, "kind", desc.kind);

      jsonFindInt(objectText, "rank", desc.rank);

      jsonFindIntArray3(objectText,
			"tile",
			desc.tileX,
			desc.tileY,
			desc.tileZ);

      jsonFindInt(objectText, "num_warps", desc.numWarps);
      jsonFindInt(objectText, "threads_per_warp", desc.threadsPerWarp);
      jsonFindInt(objectText, "num_ctas", desc.numCTAs);
      jsonFindInt(objectText, "num_stages", desc.numStages);

      if (!jsonFindInt(objectText,
		       "cuda_threads_per_cta",
		       desc.cudaThreadsPerCTA)) {
	desc.cudaThreadsPerCTA = desc.numWarps * desc.threadsPerWarp;
      }
      
      jsonFindInt(objectText, "triton_hidden_ptr_args",
		  desc.tritonHiddenPtrArgs);

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
  
  static void fngpuEnsureInitialized() {
    if (fngpuRegistry.initialized)
      return;

    const char *ptxPath =
      fngpuGetEnvOrDefault("FNGPU_PTX", "fngpu_kernels.ptx");

    const char *jsonPath =
      fngpuGetEnvOrDefault("FNGPU_KERNELS_JSON", "fngpu_kernels.json");

    if (fngpuDebugEnabled()) {
      std::fprintf(stderr, "FNGPU: loading PTX from '%s'\n", ptxPath);
      std::fprintf(stderr, "FNGPU: loading JSON from '%s'\n", jsonPath);
    }

    FNGPU_CUDA_CHECK(cuInit(0));

    CUdevice device = 0;
    FNGPU_CUDA_CHECK(cuDeviceGet(&device, 0));

    FNGPU_CUDA_CHECK(cuDevicePrimaryCtxRetain(&fngpuRegistry.context, device));
    FNGPU_CUDA_CHECK(cuCtxSetCurrent(fngpuRegistry.context));

    std::string ptx = fngpuReadTextFile(ptxPath);

    FNGPU_CUDA_CHECK(cuModuleLoadDataEx(&fngpuRegistry.module,
					ptx.c_str(),
					0,
					nullptr,
					nullptr));

    std::string json = fngpuReadTextFile(jsonPath);
    fngpuRegistry.kernels = fngpuParseKernelDescsFromJson(json);

    if (fngpuRegistry.kernels.empty()) {
      std::fprintf(stderr,
		   "FNGPU warning: no kernel descriptors parsed from '%s'\n",
		   jsonPath);
    }

    if (fngpuDebugEnabled()) {
      for (const auto &entry : fngpuRegistry.kernels) {
	const FNGPUKernelDesc &desc = entry.second;

	std::fprintf(stderr,
		     "FNGPU: registered kernel id %d -> '%s' "
		     "kind=%s rank=%d tile=(%d,%d,%d) "
		     "warps=%d threads_per_warp=%d "
		     "cuda_threads_per_cta=%d hidden_ptr_args=%d\n",
		     desc.id,
		     desc.name.c_str(),
		     desc.kind.c_str(),
		     desc.rank,
		     desc.tileX,
		     desc.tileY,
		     desc.tileZ,
		     desc.numWarps,
		     desc.threadsPerWarp,
		     desc.cudaThreadsPerCTA,
		     desc.tritonHiddenPtrArgs);
      }
    }

    fngpuRegistry.initialized = true;
  }
  
  static void fngpuEnsureCurrentContext() {
    fngpuEnsureInitialized();

    if (!fngpuRegistry.context) {
      std::fprintf(stderr, "FNGPU error: CUDA context is null\n");
      std::abort();
    }

    FNGPU_CUDA_CHECK(cuCtxSetCurrent(fngpuRegistry.context));

    if (fngpuDebugEnabled()) {
      CUcontext current = nullptr;
      FNGPU_CUDA_CHECK(cuCtxGetCurrent(&current));

      std::fprintf(stderr,
		   "FNGPU: current CUDA context = %p, registry context = %p\n",
		   static_cast<void *>(current),
		   static_cast<void *>(fngpuRegistry.context));
    }
  }
  
  static const FNGPUKernelDesc *fngpuLookupKernelDesc(int32_t kernelId) {
    fngpuEnsureCurrentContext();

    auto it = fngpuRegistry.kernels.find(kernelId);
    if (it == fngpuRegistry.kernels.end())
      return nullptr;

    return &it->second;
  }

  static int32_t fngpuTritonHiddenPtrArgCount(int32_t kernelId) {
    if (const FNGPUKernelDesc *desc = fngpuLookupKernelDesc(kernelId))
      return desc->tritonHiddenPtrArgs;

    return 2;
  }

  static void fngpuValidateSupportedHiddenPtrArgCount(int32_t kernelId) {
    int32_t count = fngpuTritonHiddenPtrArgCount(kernelId);

    if (count == 2)
      return;

    std::fprintf(stderr,
		 "FNGPU error: kernel id %d requires %d Triton hidden pointer "
		 "args, but this runtime currently supports exactly 2\n",
		 kernelId,
		 count);
    std::abort();
  }

  // -------------------------------------------------------------------------- //
  // CUDA module/function management
  // -------------------------------------------------------------------------- //
  static void fngpuDebugFunctionAttributes(CUfunction fn, int32_t kernelId) {
    if (!fngpuDebugEnabled())
      return;

    int maxThreadsPerBlock = 0;
    int numRegs = 0;
    int sharedBytes = 0;
    int binaryVersion = 0;
    int ptxVersion = 0;

    cuFuncGetAttribute(&maxThreadsPerBlock,
		       CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
		       fn);

    cuFuncGetAttribute(&numRegs,
		       CU_FUNC_ATTRIBUTE_NUM_REGS,
		       fn);

    cuFuncGetAttribute(&sharedBytes,
		       CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
		       fn);

    cuFuncGetAttribute(&binaryVersion,
		       CU_FUNC_ATTRIBUTE_BINARY_VERSION,
		       fn);

    cuFuncGetAttribute(&ptxVersion,
		       CU_FUNC_ATTRIBUTE_PTX_VERSION,
		       fn);

    std::fprintf(stderr,
		 "FNGPU: function attrs for kernel id %d: "
		 "max_threads_per_block=%d num_regs=%d shared_bytes=%d "
		 "binary_version=%d ptx_version=%d\n",
		 kernelId,
		 maxThreadsPerBlock,
		 numRegs,
		 sharedBytes,
		 binaryVersion,
		 ptxVersion);
  }
  
  static CUfunction getKernelFunction(int32_t kernelId) {
    fngpuEnsureCurrentContext();

    auto cacheIt = fngpuRegistry.functionCache.find(kernelId);
    if (cacheIt != fngpuRegistry.functionCache.end())
      return cacheIt->second;

    std::string kernelName;

    if (const FNGPUKernelDesc *desc = fngpuLookupKernelDesc(kernelId)) {
      kernelName = desc->name;
    } else {
      kernelName = "fngpu_kernel_" + std::to_string(kernelId);

      std::fprintf(stderr,
		   "FNGPU warning: no JSON descriptor for kernel id %d; "
		   "falling back to symbol name '%s'\n",
		   kernelId,
		   kernelName.c_str());
    }

    if (fngpuDebugEnabled()) {
      std::fprintf(stderr,
		   "FNGPU: looking up CUDA kernel id %d as symbol '%s'\n",
		   kernelId,
		   kernelName.c_str());
    }

    CUfunction fn = nullptr;
    FNGPU_CUDA_CHECK(cuModuleGetFunction(&fn,
					 fngpuRegistry.module,
					 kernelName.c_str()));

    fngpuRegistry.functionCache[kernelId] = fn;
    return fn;
  }

  static unsigned fngpuCudaThreadsPerCTA(int32_t kernelId) {
    if (const FNGPUKernelDesc *desc = fngpuLookupKernelDesc(kernelId)) {
      if (desc->cudaThreadsPerCTA > 0)
	return static_cast<unsigned>(desc->cudaThreadsPerCTA);

      if (desc->numWarps > 0 && desc->threadsPerWarp > 0)
	return static_cast<unsigned>(desc->numWarps * desc->threadsPerWarp);
    }

    return 32;
  }

  static void fngpuValidateHostLaunchAgainstDesc(int32_t kernelId,
						 int32_t rank,
						 int32_t blockX,
						 int32_t blockY,
						 int32_t blockZ) {
    const FNGPUKernelDesc *desc = fngpuLookupKernelDesc(kernelId);
    if (!desc)
      return;

    if (desc->rank != rank) {
      std::fprintf(stderr,
		   "FNGPU warning: host launch rank %d disagrees with JSON "
		   "rank %d for kernel id %d\n",
		   rank,
		   desc->rank,
		   kernelId);
    }

    if (desc->tileX != blockX || desc->tileY != blockY ||
	desc->tileZ != blockZ) {
      std::fprintf(stderr,
		   "FNGPU warning: host tile (%d,%d,%d) disagrees with JSON "
		   "tile (%d,%d,%d) for kernel id %d\n",
		   blockX,
		   blockY,
		   blockZ,
		   desc->tileX,
		   desc->tileY,
		   desc->tileZ,
		   kernelId);
    }
  }

  static unsigned fngpuCdiv(int32_t x, int32_t y) {
    return static_cast<unsigned>((x + y - 1) / y);
  }

  static std::size_t fngpuElementCount(int32_t rank,
				       int32_t extentX,
				       int32_t extentY,
				       int32_t extentZ) {
    std::size_t count = static_cast<std::size_t>(extentX);

    if (rank >= 2)
      count *= static_cast<std::size_t>(extentY);

    if (rank >= 3)
      count *= static_cast<std::size_t>(extentZ);

    return count;
  }

  static void fngpuValidateCommonLaunchInputs(const char *entryName,
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
    if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
      std::fprintf(stderr,
		   "FNGPU error: invalid tile/block shape (%d,%d,%d) in %s\n",
		   blockX,
		   blockY,
		   blockZ,
		   entryName);
      std::abort();
    }

    if (rank < 1 || rank > 3) {
      std::fprintf(stderr,
		   "FNGPU error: unsupported rank %d in %s\n",
		   rank,
		   entryName);
      std::abort();
    }

    if (!a || !b || !c) {
      std::fprintf(stderr,
		   "FNGPU error: null host pointer in %s: "
		   "a=%p b=%p c=%p\n",
		   entryName,
		   static_cast<void *>(a),
		   static_cast<void *>(b),
		   static_cast<void *>(c));
      std::abort();
    }

    if (extentX <= 0 || extentY <= 0 || extentZ <= 0) {
      if (fngpuDebugEnabled()) {
	std::fprintf(stderr,
		     "FNGPU: non-positive extent (%d,%d,%d) in %s; "
		     "skipping launch\n",
		     extentX,
		     extentY,
		     extentZ,
		     entryName);
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
  fngpuEnsureCurrentContext();
 
  fngpuValidateHostLaunchAgainstDesc(kernelId,
                                     rank,
                                     blockX,
                                     blockY,
                                     blockZ);

  if (rank != 1 && rank != 2) {
    std::fprintf(stderr,
                 "FNGPU error: __fngpu_launch_nd_f32 currently supports "
                 "only rank 1 or 2, got rank %d\n",
                 rank);
    std::abort();
  }

  fngpuValidateCommonLaunchInputs("__fngpu_launch_nd_f32",
                                  rank,
                                  blockX,
                                  blockY,
                                  blockZ,
                                  a,
                                  b,
                                  c,
                                  extentX,
                                  extentY,
                                  extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);
  fngpuDebugFunctionAttributes(fn, kernelId);

  unsigned gridX = fngpuCdiv(extentX, blockX);
  unsigned gridY = rank >= 2 ? fngpuCdiv(extentY, blockY) : 1;
  unsigned gridZ = 1;

  unsigned cudaBlockX = fngpuCudaThreadsPerCTA(kernelId);

  std::size_t elemCount =
    fngpuElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes = elemCount * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNGPU_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
                 "FNGPU: launch binary kernel id=%d rank=%d "
                 "grid=(%u,%u,%u) tile=(%d,%d,%d) "
                 "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
                 kernelId,
                 rank,
                 gridX,
                 gridY,
                 gridZ,
                 blockX,
                 blockY,
                 blockZ,
                 cudaBlockX,
                 extentX,
                 extentY,
                 extentZ,
                 numBytes);
  }

  if (rank == 1) {
    fngpuValidateSupportedHiddenPtrArgCount(kernelId);

    FNGPUHiddenTritonArgs hidden;

    void *args[] = {
      &dA,
      &dB,
      &dC,
      &extentX,
      &hidden.hidden0,
      &hidden.hidden1,
    };

    if (fngpuDebugEnabled()) {
      std::fprintf(stderr,
		   "FNGPU: binary rank1 args: "
		   "dA=0x%llx dB=0x%llx dC=0x%llx extentX=%d "
		   "hidden0=0x%llx hidden1=0x%llx "
		   "args={%p,%p,%p,%p,%p,%p}\n",                 
		   static_cast<unsigned long long>(dA),
		   static_cast<unsigned long long>(dB),
		   static_cast<unsigned long long>(dC),
		   extentX,
		   static_cast<unsigned long long>(hidden.hidden0),
		   static_cast<unsigned long long>(hidden.hidden1),
		   args[0],
		   args[1],
		   args[2],
		   args[3],
		   args[4],
		   args[5]);

      std::fprintf(stderr, "FNGPU: about to cuLaunchKernel rank1\n");
      std::fflush(stderr);
    }

    FNGPU_CUDA_CHECK(cuLaunchKernel(fn,
                                    gridX, 1, 1,
                                    cudaBlockX, 1, 1,
                                    0,
                                    nullptr,
                                    args,
                                    nullptr));
    if (fngpuDebugEnabled()) {
      std::fprintf(stderr, "FNGPU: cuLaunchKernel rank1 returned\n");
      std::fflush(stderr);
    }

  } else {
    fngpuValidateSupportedHiddenPtrArgCount(kernelId);
    
    FNGPUHiddenTritonArgs hidden;

    void *args[] = {
      &dA,
      &dB,
      &dC,
      &extentX,
      &extentY,
      &hidden.hidden0,
      &hidden.hidden1,
    };

    if (fngpuDebugEnabled()) {
      std::fprintf(stderr,
		   "FNGPU: binary rank2 args: "
		   "dA=0x%llx dB=0x%llx dC=0x%llx "
		   "extentX=%d extentY=%d "
		   "hidden0=0x%llx hidden1=0x%llx "
		   "args={%p,%p,%p,%p,%p,%p,%p}\n",
		   static_cast<unsigned long long>(dA),
		   static_cast<unsigned long long>(dB),
		   static_cast<unsigned long long>(dC),
		   extentX,
		   extentY,
		   static_cast<unsigned long long>(hidden.hidden0),
		   static_cast<unsigned long long>(hidden.hidden1),
		   args[0],
		   args[1],
		   args[2],
		   args[3],
		   args[4],
		   args[5],
		   args[6]);

      std::fprintf(stderr, "FNGPU: about to cuLaunchKernel rank2\n");
      std::fflush(stderr);
    }

    FNGPU_CUDA_CHECK(cuLaunchKernel(fn,
                                    gridX, gridY, 1,
                                    cudaBlockX, 1, 1,
                                    0,
                                    nullptr,
                                    args,
                                    nullptr));
    if (fngpuDebugEnabled()) {
      std::fprintf(stderr, "FNGPU: cuLaunchKernel rank2 returned\n");
      std::fflush(stderr);
    }
  }

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr, "FNGPU: about to cuCtxSynchronize\n");
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuCtxSynchronize());

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr, "FNGPU: cuCtxSynchronize returned\n");
    std::fflush(stderr);
  }

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
		 "FNGPU: about to cuMemcpyDtoH c=%p dC=0x%llx bytes=%zu\n",
		 static_cast<void *>(c),
		 static_cast<unsigned long long>(dC),
		 numBytes);
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr, "FNGPU: cuMemcpyDtoH returned\n");
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuMemFree(dA));
  FNGPU_CUDA_CHECK(cuMemFree(dB));
  FNGPU_CUDA_CHECK(cuMemFree(dC));
}

// -------------------------------------------------------------------------- //
// Public runtime ABI: 1-scalar f32 SAXPY-style kernels
// -------------------------------------------------------------------------- //
//
// Current SAXPY TTIR signature:
//
//   (%a: ptr<f32>, %b: ptr<f32>, %c: ptr<f32>, %alpha: f32, %n: i32)
//

extern "C" void __fngpu_launch_nd_f32_s1(
					 int32_t kernelId,
					 int32_t rank,
					 int32_t blockX,
					 int32_t blockY,
					 int32_t blockZ,
					 float *a,
					 float *b,
					 float *c,
					 float scalar0,
					 int32_t extentX,
					 int32_t extentY,
					 int32_t extentZ) {
  fngpuEnsureCurrentContext();

  fngpuValidateHostLaunchAgainstDesc(kernelId,
                                     rank,
                                     blockX,
                                     blockY,
                                     blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
                 "FNGPU error: __fngpu_launch_nd_f32_s1 currently supports "
                 "only rank 1, got rank %d\n",
                 rank);
    std::abort();
  }

  fngpuValidateCommonLaunchInputs("__fngpu_launch_nd_f32_s1",
                                  rank,
                                  blockX,
                                  blockY,
                                  blockZ,
                                  a,
                                  b,
                                  c,
                                  extentX,
                                  extentY,
                                  extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fngpuCdiv(extentX, blockX);
  unsigned cudaBlockX = fngpuCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes = numElems * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNGPU_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  FNGPUHiddenTritonArgs hidden;

  void *args[] = {
    &dA,
    &dB,
    &dC,
    &scalar0,
    &extentX,
    &hidden.hidden0,
    &hidden.hidden1,
  };

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
                 "FNGPU: launch SAXPY kernel id=%d rank=%d "
                 "grid=(%u,1,1) tile=(%d,%d,%d) "
                 "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
                 "scalar0=%f bytes=%zu\n",
                 kernelId,
                 rank,
                 gridX,
                 blockX,
                 blockY,
                 blockZ,
                 cudaBlockX,
                 extentX,
                 extentY,
                 extentZ,
                 static_cast<double>(scalar0),
                 numBytes);
  }


  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
		 "FNGPU: SAXPY args: "
		 "dA=0x%llx dB=0x%llx dC=0x%llx "
		 "scalar0=%f extentX=%d "
		 "hidden0=0x%llx hidden1=0x%llx "
		 "args={%p,%p,%p,%p,%p,%p,%p}\n",
		 static_cast<unsigned long long>(dA),
		 static_cast<unsigned long long>(dB),
		 static_cast<unsigned long long>(dC),
		 static_cast<double>(scalar0),
		 extentX,
		 static_cast<unsigned long long>(hidden.hidden0),
		 static_cast<unsigned long long>(hidden.hidden1),
		 args[0],
		 args[1],
		 args[2],
		 args[3],
		 args[4],
		 args[5],
		 args[6]);

    std::fprintf(stderr, "FNGPU: about to cuLaunchKernel SAXPY\n");
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuLaunchKernel(fn,
				  gridX, 1, 1,
				  cudaBlockX, 1, 1,
				  0,
				  nullptr,
				  args,
				  nullptr));

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr, "FNGPU: cuLaunchKernel SAXPY returned\n");
    std::fflush(stderr);
  }


  FNGPU_CUDA_CHECK(cuCtxSynchronize());

  FNGPU_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  FNGPU_CUDA_CHECK(cuMemFree(dA));
  FNGPU_CUDA_CHECK(cuMemFree(dB));
  FNGPU_CUDA_CHECK(cuMemFree(dC));
}

extern "C" void __fngpu_launch_nd_f32_s2(
    int32_t kernelId,
    int32_t rank,
    int32_t blockX,
    int32_t blockY,
    int32_t blockZ,
    float *a,
    float *b,
    float *c,
    float scalar0,
    float scalar1,
    int32_t extentX,
    int32_t extentY,
    int32_t extentZ) {
  fngpuEnsureCurrentContext();

  fngpuValidateHostLaunchAgainstDesc(kernelId,
                                     rank,
                                     blockX,
                                     blockY,
                                     blockZ);

  if (rank != 1) {
    std::fprintf(stderr,
                 "FNGPU error: __fngpu_launch_nd_f32_s2 currently supports "
                 "only rank 1, got rank %d\n",
                 rank);
    std::abort();
  }

  fngpuValidateCommonLaunchInputs("__fngpu_launch_nd_f32_s2",
                                  rank,
                                  blockX,
                                  blockY,
                                  blockZ,
                                  a,
                                  b,
                                  c,
                                  extentX,
                                  extentY,
                                  extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fngpuCdiv(extentX, blockX);
  unsigned cudaBlockX = fngpuCudaThreadsPerCTA(kernelId);

  std::size_t numElems = static_cast<std::size_t>(extentX);
  std::size_t numBytes = numElems * sizeof(float);

  CUdeviceptr dA = 0;
  CUdeviceptr dB = 0;
  CUdeviceptr dC = 0;

  FNGPU_CUDA_CHECK(cuMemAlloc(&dA, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dB, numBytes));
  FNGPU_CUDA_CHECK(cuMemAlloc(&dC, numBytes));

  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dA, a, numBytes));
  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dB, b, numBytes));

  fngpuValidateSupportedHiddenPtrArgCount(kernelId);

  FNGPUHiddenTritonArgs hidden;

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

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
                 "FNGPU: launch expr/s2 kernel id=%d rank=%d "
                 "grid=(%u,1,1) tile=(%d,%d,%d) "
                 "cuda_block=(%u,1,1) extent=(%d,%d,%d) "
                 "scalar0=%f scalar1=%f bytes=%zu\n",
                 kernelId,
                 rank,
                 gridX,
                 blockX,
                 blockY,
                 blockZ,
                 cudaBlockX,
                 extentX,
                 extentY,
                 extentZ,
                 static_cast<double>(scalar0),
                 static_cast<double>(scalar1),
                 numBytes);

    std::fprintf(stderr,
                 "FNGPU: s2 args: "
                 "dA=0x%llx dB=0x%llx dC=0x%llx "
                 "scalar0=%f scalar1=%f extentX=%d "
                 "hidden0=0x%llx hidden1=0x%llx "
                 "args={%p,%p,%p,%p,%p,%p,%p,%p}\n",
                 static_cast<unsigned long long>(dA),
                 static_cast<unsigned long long>(dB),
                 static_cast<unsigned long long>(dC),
                 static_cast<double>(scalar0),
                 static_cast<double>(scalar1),
                 extentX,
                 static_cast<unsigned long long>(hidden.hidden0),
                 static_cast<unsigned long long>(hidden.hidden1),
                 args[0],
                 args[1],
                 args[2],
                 args[3],
                 args[4],
                 args[5],
                 args[6],
                 args[7]);

    std::fprintf(stderr, "FNGPU: about to cuLaunchKernel s2\n");
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuLaunchKernel(fn,
                                  gridX, 1, 1,
                                  cudaBlockX, 1, 1,
                                  0,
                                  nullptr,
                                  args,
                                  nullptr));

  if (fngpuDebugEnabled()) {
    std::fprintf(stderr, "FNGPU: cuLaunchKernel s2 returned\n");
    std::fflush(stderr);
  }

  FNGPU_CUDA_CHECK(cuCtxSynchronize());

  FNGPU_CUDA_CHECK(cuMemcpyDtoH(c, dC, numBytes));

  FNGPU_CUDA_CHECK(cuMemFree(dA));
  FNGPU_CUDA_CHECK(cuMemFree(dB));
  FNGPU_CUDA_CHECK(cuMemFree(dC));
}


extern "C" void __fngpu_launch_f32_v1(
    int32_t kernelId,
    int32_t rank,
    int32_t blockX,
    int32_t blockY,
    int32_t blockZ,
    int32_t numReadArrays,
    int32_t numScalars,
    float *read0,
    float *read1,
    float *read2,
    float *write,
    float scalar0,
    float scalar1,
    float scalar2,
    int32_t extentX,
    int32_t extentY,
    int32_t extentZ) {
  fngpuEnsureCurrentContext();

  fngpuValidateHostLaunchAgainstDesc(kernelId,
                                     rank,
                                     blockX,
                                     blockY,
                                     blockZ);

  if (numReadArrays < 1 || numReadArrays > 3) {
    std::fprintf(stderr,
                 "FNGPU error: unsupported numReadArrays=%d for kernel id %d\n",
                 numReadArrays,
                 kernelId);
    std::abort();
  }

  if (numScalars < 0 || numScalars > 3) {
    std::fprintf(stderr,
                 "FNGPU error: unsupported numScalars=%d for kernel id %d\n",
                 numScalars,
                 kernelId);
    std::abort();
  }

  fngpuValidateCommonLaunchInputs("__fngpu_launch_f32_v1",
                                  rank,
                                  blockX,
                                  blockY,
                                  blockZ,
                                  read0,
                                  read1,
                                  write,
                                  extentX,
                                  extentY,
                                  extentZ);

  if (extentX <= 0 || extentY <= 0 || extentZ <= 0)
    return;

  CUfunction fn = getKernelFunction(kernelId);

  unsigned gridX = fngpuCdiv(extentX, blockX);
  unsigned gridY = rank >= 2 ? fngpuCdiv(extentY, blockY) : 1;
  unsigned cudaBlockX = fngpuCudaThreadsPerCTA(kernelId);

  std::size_t elemCount =
      fngpuElementCount(rank, extentX, extentY, extentZ);

  std::size_t numBytes = elemCount * sizeof(float);

  CUdeviceptr dRead0 = 0;
  CUdeviceptr dRead1 = 0;
  CUdeviceptr dRead2 = 0;
  CUdeviceptr dWrite = 0;

  FNGPU_CUDA_CHECK(cuMemAlloc(&dRead0, numBytes));
  FNGPU_CUDA_CHECK(cuMemcpyHtoD(dRead0, read0, numBytes));

  if (numReadArrays >= 2) {
    FNGPU_CUDA_CHECK(cuMemAlloc(&dRead1, numBytes));
    FNGPU_CUDA_CHECK(cuMemcpyHtoD(dRead1, read1, numBytes));
  }

  if (numReadArrays >= 3) {
    FNGPU_CUDA_CHECK(cuMemAlloc(&dRead2, numBytes));
    FNGPU_CUDA_CHECK(cuMemcpyHtoD(dRead2, read2, numBytes));
  }

  FNGPU_CUDA_CHECK(cuMemAlloc(&dWrite, numBytes));

  fngpuValidateSupportedHiddenPtrArgCount(kernelId);
  FNGPUHiddenTritonArgs hidden;

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
               "FNGPU error: internal runtime argument buffer overflow "
               "for kernel id %d; argCount=%d\n",
               kernelId,
               argCount);
  std::abort();
}


  if (fngpuDebugEnabled()) {
    std::fprintf(stderr,
                 "FNGPU: launch generic f32 kernel id=%d rank=%d "
                 "reads=%d scalars=%d grid=(%u,%u,1) tile=(%d,%d,%d) "
                 "cuda_block=(%u,1,1) extent=(%d,%d,%d) bytes=%zu\n",
                 kernelId,
                 rank,
                 numReadArrays,
                 numScalars,
                 gridX,
                 gridY,
                 blockX,
                 blockY,
                 blockZ,
                 cudaBlockX,
                 extentX,
                 extentY,
                 extentZ,
                 numBytes);
  }

   FNGPU_CUDA_CHECK(cuLaunchKernel(fn,
                                gridX,
                                gridY,
                                1,
                                cudaBlockX,
                                1,
                                1,
                                0,
                                nullptr,
                                args,
                                nullptr));



  FNGPU_CUDA_CHECK(cuCtxSynchronize());

  FNGPU_CUDA_CHECK(cuMemcpyDtoH(write, dWrite, numBytes));

  FNGPU_CUDA_CHECK(cuMemFree(dRead0));

  if (dRead1)
    FNGPU_CUDA_CHECK(cuMemFree(dRead1));

  if (dRead2)
    FNGPU_CUDA_CHECK(cuMemFree(dRead2));

  FNGPU_CUDA_CHECK(cuMemFree(dWrite));
}


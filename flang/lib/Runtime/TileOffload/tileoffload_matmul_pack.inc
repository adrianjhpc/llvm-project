// Internal implementation included after the variadic ABI helpers.
// The compiler emits candidates only for ordinary TF32 matmul with unit steps.
static const TileOffloadKernelDesc *TileOffloadTryPackMatmulA(
    const TileOffloadKernelDesc *desc,
    const TileOffloadPendingLaunchV2 &pending,
    const std::vector<CUdeviceptr> &devicePointers,
    std::vector<int32_t> &parameterValues, std::vector<void *> &arguments) {
  if (desc->autoPackAId < 0 || desc->autoPackedMatmulId < 0)
    return nullptr;
  const char *policy = std::getenv("TILEOFF_MATMUL_PACK_A");
  if (policy && std::strcmp(policy, "0") == 0)
    return nullptr;
  bool force = policy && std::strcmp(policy, "1") == 0;
  if (policy && !force && std::strcmp(policy, "auto") != 0) {
    std::fprintf(stderr,
        "TileOffload error: TILEOFF_MATMUL_PACK_A must be 0, 1 or auto\n");
    std::abort();
  }
  int32_t m = pending.extent[0], n = pending.extent[1], k = pending.extent[2];
  // Provisional conservative crossover, not a hardware-independent cost model.
  if (m <= 0 || n <= 0 || k <= 0 ||
      (!force && (m < 512 || n < 512 || k < 512)) ||
      k > std::numeric_limits<int32_t>::max() - 15)
    return nullptr;
  if (!desc->isMatmul || desc->matmulTransposedA || desc->backend != "triton" ||
      desc->acceleratorTarget != "cuda" || desc->parameters.size() != 21 ||
      pending.arrays.size() != 3 || desc->loopStep[0] != 1 ||
      desc->loopStep[1] != 1 || desc->loopStep[2] != 1)
    return nullptr;
  int aIndex = desc->parameters[0].arrayIndex;
  int bIndex = desc->parameters[1].arrayIndex;
  int cIndex = desc->parameters[2].arrayIndex;
  if (aIndex < 0 || aIndex >= 3 || bIndex < 0 || bIndex >= 3 || cIndex < 0 ||
      cIndex >= 3 || aIndex == bIndex || aIndex == cIndex || bIndex == cIndex ||
      desc->parameters[0].type != "ptr<f32>")
    return nullptr;
  // Host ranges can overlap even when FIR arguments have distinct identities.
  // Retain original behavior for aliased inputs/outputs and exotic descriptors.
  for (int i = 0; i < 3; ++i) {
    const auto &a = pending.arrays[i];
    if (a.stride[0] != 1 || a.stride[1] <= 0)
      return nullptr;
    uintptr_t begin = reinterpret_cast<uintptr_t>(a.host);
    if (a.bytes > std::numeric_limits<uintptr_t>::max() - begin)
      return nullptr;
    for (int j = 0; j < i; ++j) {
      const auto &b = pending.arrays[j];
      uintptr_t other = reinterpret_cast<uintptr_t>(b.host);
      if (begin < other + b.bytes && other < begin + a.bytes)
        return nullptr;
    }
  }
  const auto *packDesc = TileOffloadLookupKernelDesc(desc->autoPackAId);
  const auto *packedDesc =
      TileOffloadLookupKernelDesc(desc->autoPackedMatmulId);
  bool valid = packDesc && packedDesc && packDesc->kind == "matmul_pack_a" &&
      packDesc->backend == "triton" && packDesc->acceleratorTarget == "cuda" &&
      packDesc->cudaThreadsPerCTA == 128 &&
      packDesc->tritonHiddenPtrArgs == 2 &&
      packDesc->compiledSharedBytes >= 0 &&
      packedDesc->compiledSharedBytes >= 0 && packedDesc->isMatmul &&
      packedDesc->matmulTransposedA && packedDesc->backend == desc->backend &&
      packedDesc->acceleratorTarget == desc->acceleratorTarget &&
      packedDesc->cudaThreadsPerCTA == desc->cudaThreadsPerCTA &&
      packedDesc->tritonHiddenPtrArgs == desc->tritonHiddenPtrArgs &&
      packedDesc->tileX == desc->tileX && packedDesc->tileY == desc->tileY &&
      packedDesc->tileZ == desc->tileZ &&
      packedDesc->parameters.size() == desc->parameters.size();
  if (valid) {
    for (size_t i = 0; i < desc->parameters.size(); ++i) {
      const auto &a = desc->parameters[i];
      const auto &b = packedDesc->parameters[i];
      valid &= a.slot == b.slot && a.role == b.role && a.type == b.type &&
          a.arrayIndex == b.arrayIndex && a.dimension == b.dimension;
    }
  }
  if (!valid) {
    std::fprintf(stderr,
        "TileOffload error: incomplete automatic matmul packing metadata; "
        "rebuild with the updated driver (kernel id=%d)\n",
        desc->id);
    std::abort();
  }
  int32_t ld = (k + 15) & ~15;
  size_t bytes =
      TileOffloadCheckedMul(TileOffloadCheckedMul(static_cast<size_t>(m),
                                static_cast<size_t>(ld), "packed A elements"),
          sizeof(float), "packed A bytes");
  auto &state = TileOffloadActiveContextState();
  auto &workspace = state.matmulPackWorkspace;
  if (workspace.bytes < bytes) {
    // Allocate first so allocation failure leaves both the original path and
    // existing workspace usable. Synchronize only when replacing old storage.
    CUdeviceptr replacement = 0;
    CUresult status = cuMemAlloc(&replacement, bytes);
    if (status != CUDA_SUCCESS) {
      if (status != CUDA_ERROR_OUT_OF_MEMORY) {
        TILEOFF_CUDA_CHECK(status);
      }
      if (TileOffloadDebugEnabled())
        std::fprintf(stderr,
            "TileOffload: packed A allocation unavailable; using original "
            "matmul\n");
      return nullptr;
    }
    if (workspace.ptr) {
      TileOffloadSynchronizeActiveContext();
      TILEOFF_CUDA_CHECK(cuMemFree(workspace.ptr));
    }
    workspace.ptr = replacement;
    workspace.bytes = bytes;
  }
  const auto &a = pending.arrays[aIndex];
  int64_t sourceLD = a.stride[1];
  int64_t base = (int64_t(pending.loopLower[0]) - a.lower[0]) +
      (int64_t(pending.loopLower[2]) - a.lower[1]) * sourceLD;
  CUdeviceptr source = devicePointers[aIndex];
  TileOffloadHiddenTritonArgs hidden;
  void *packArgs[] = {&source, &workspace.ptr, &m, &k, &ld, &base, &sourceLD,
      &hidden.hidden0, &hidden.hidden1};
  CUfunction packFunction = getKernelFunction(packDesc->id);
  unsigned shared = static_cast<unsigned>(packDesc->compiledSharedBytes);
  TileOffloadConfigureDynamicSharedMemory(packFunction, packDesc->id, shared);
  TILEOFF_CUDA_CHECK(
      cuLaunchKernel(packFunction, TileOffloadCdiv(m, 32, "pack A rows"),
          TileOffloadCdiv(k, 32, "pack A columns"), 1, 128, 1, 1, shared,
          state.stream, packArgs, nullptr));
  // AP stores (k,i); source loop coordinates and all B/C arguments are
  // retained.
  arguments[0] = &workspace.ptr;
  for (const auto &parameter : desc->parameters) {
    if (parameter.arrayIndex != aIndex)
      continue;
    if (parameter.role == TileOffloadKernelParameterRole::ArrayLowerBound)
      parameterValues[parameter.slot] =
          pending.loopLower[parameter.dimension == 0 ? 2 : 0];
    else if (parameter.role == TileOffloadKernelParameterRole::ArrayStride)
      parameterValues[parameter.slot] = parameter.dimension == 0 ? 1 : ld;
  }
  if (TileOffloadDebugEnabled())
    std::fprintf(stderr,
        "TileOffload: auto-pack A kernel=%d pack=%d packed=%d "
        "M=%d N=%d K=%d ld=%d bytes=%zu\n",
        desc->id, packDesc->id, packedDesc->id, m, n, k, ld, bytes);
  return packedDesc;
}

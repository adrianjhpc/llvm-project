#ifndef FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_KERNEL_PLAN_H
#define FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_KERNEL_PLAN_H

#include "flang/Optimizer/Dialect/TileOffload/TileOffloadKernelAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace fir::TileOffload {

/// Compiler-side device IR produced by an TileOffload code-generation backend.
/// These values describe intermediate artifacts, not necessarily something a
/// runtime loader can consume directly.
enum class TileOffloadDeviceIRKind { TTIR, LLVMIR, CUDATileIR, PTX };

/// Runtime-consumable device image formats.
enum class TileOffloadDeviceImageKind { PTX, Cubin, HSACO };

inline llvm::StringRef TileOffloadDeviceIRKindName(TileOffloadDeviceIRKind kind) {
  switch (kind) {
  case TileOffloadDeviceIRKind::TTIR:
    return "ttir";
  case TileOffloadDeviceIRKind::LLVMIR:
    return "llvm-ir";
  case TileOffloadDeviceIRKind::CUDATileIR:
    return "cuda-tile-ir";
  case TileOffloadDeviceIRKind::PTX:
    return "ptx";
  }
  return "unknown";
}

inline llvm::StringRef TileOffloadDeviceImageKindName(TileOffloadDeviceImageKind kind) {
  switch (kind) {
  case TileOffloadDeviceImageKind::PTX:
    return "ptx";
  case TileOffloadDeviceImageKind::Cubin:
    return "cubin";
  case TileOffloadDeviceImageKind::HSACO:
    return "hsaco";
  }
  return "unknown";
}

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
  ArrayStride
};

enum class TileOffloadKernelParameterPassing { DevicePointer, Value };

/// One source-level parameter in the stable TileOffload kernel ABI. Backend-private
/// parameters, such as parameters appended by Triton/NVVM, are deliberately
/// not represented here.
struct TileOffloadKernelParameter {
  unsigned slot = 0;
  TileOffloadKernelParameterRole role = TileOffloadKernelParameterRole::Read;
  TileOffloadKernelParameterPassing passing = TileOffloadKernelParameterPassing::Value;
  ElementType elementType = ElementType::Unknown;
  std::string name;
  int32_t arrayIndex = -1;
  int32_t scalarIndex = -1;
  int32_t dimension = -1;
};

struct TileOffloadPackBinding {
  unsigned kernelArgSlot = 0;
  int32_t target = 0;
};

struct TileOffloadKernelABI {
  llvm::SmallVector<TileOffloadKernelParameter> parameters;
  llvm::SmallVector<TileOffloadPackBinding> packBindings;
};

struct TileOffloadTileShape {
  int64_t x = 1;
  int64_t y = 1;
  int64_t z = 1;
};

enum class TileOffloadMatmulStrategy { Dot, Reduce, FMA };

/// Scheduling requests expressed without naming a particular backend. A
/// backend may reject a schedule or map it to its closest native concept.
struct TileOffloadKernelSchedule {
  TileOffloadTileShape tile;
  int32_t parallelSubgroups = 1;
  int32_t subgroupWidth = 32;
  int32_t pipelineStages = 3;
  TileOffloadMatmulStrategy f64MatmulStrategy = TileOffloadMatmulStrategy::Reduce;
};

struct TileOffloadReductionStagePlan {
  int32_t id = -1;
  std::string name;
  ReductionOperator reductionOperator = ReductionOperator::Add;
  ElementType elementType = ElementType::Unknown;
  TileOffloadKernelABI abi;
};

/// Backend-neutral description of one recognized TileOffload launch.
///
/// ElementwiseKernel retains the recognized FIR values and expression tree.
/// The remaining fields contain stable identity, schedule and ABI information
/// that used to be reconstructed inside the Triton emitter.
struct TileOffloadKernelPlan {
  fir::TileOffload::LaunchOp launchOp;
  int32_t id = -1;
  std::string name;
  ElementwiseKernel kernel;
  bool usesVariadicABI = false;
  bool copyBackWrites = true;
  TileOffloadKernelSchedule schedule;
  TileOffloadKernelABI abi;
  std::optional<TileOffloadReductionStagePlan> reductionStage;
};

struct TileOffloadKernelPlanOptions {
  int32_t requestedParallelSubgroups = 1;
  int32_t subgroupWidth = 32;
  int32_t pipelineStages = 3;
  TileOffloadMatmulStrategy f64MatmulStrategy = TileOffloadMatmulStrategy::Reduce;
};

class TileOffloadKernelPlanResult {
public:
  static TileOffloadKernelPlanResult success(TileOffloadKernelPlan plan);
  static TileOffloadKernelPlanResult failure(mlir::Operation *where,
                                       std::string reason);

  bool succeeded() const { return plan.has_value(); }
  bool failed() const { return !succeeded(); }

  const TileOffloadKernelPlan &getPlan() const { return *plan; }
  TileOffloadKernelPlan takePlan();
  const RecognitionFailure &getFailure() const { return failureInfo; }

private:
  std::optional<TileOffloadKernelPlan> plan;
  RecognitionFailure failureInfo;
};

TileOffloadKernelPlanResult
buildTileOffloadKernelPlan(fir::TileOffload::LaunchOp launchOp, int32_t fallbackId,
                     int32_t nextSyntheticKernelId,
                     const TileOffloadKernelPlanOptions &options);

bool isReductionKernelKind(ElementwiseKernelKind kind);
llvm::StringRef TileOffloadKernelKindName(ElementwiseKernelKind kind);

struct TileOffloadBackendSupport {
  bool supported = false;
  std::string reason;

  static TileOffloadBackendSupport success() { return {true, {}}; }
  static TileOffloadBackendSupport failure(llvm::StringRef reason) {
    return {false, reason.str()};
  }
};

/// Interface shared by TileOffload device-code backends. Module framing and kernel
/// emission use raw_ostream so textual and bytecode backends can share the
/// orchestration layer.
class TileOffloadCodegenBackend {
public:
  virtual ~TileOffloadCodegenBackend() = default;

  virtual llvm::StringRef getName() const = 0;
  virtual llvm::StringRef getAcceleratorTarget() const = 0;
  virtual TileOffloadDeviceIRKind getDeviceIRKind() const {
    return TileOffloadDeviceIRKind::TTIR;
  }
  virtual TileOffloadDeviceImageKind getRuntimeImageKind() const = 0;
  virtual TileOffloadBackendSupport
  querySupport(const TileOffloadKernelPlan &plan) const = 0;

  virtual void beginModule(const TileOffloadKernelPlanOptions &options,
                           llvm::raw_ostream &os) const = 0;
  virtual mlir::LogicalResult emitKernel(const TileOffloadKernelPlan &plan,
                                         llvm::raw_ostream &os) const = 0;
  virtual void endModule(llvm::raw_ostream &os) const = 0;

  /// Number of backend-private pointer arguments appended after the stable
  /// TileOffload ABI. This preserves compatibility with the current runtime while
  /// keeping those arguments out of TileOffloadKernelABI.
  virtual int32_t
  getPrivatePointerArgumentCount(const TileOffloadKernelPlan &plan) const = 0;
};

struct TileOffloadBackendSelection {
  const TileOffloadCodegenBackend *backend = nullptr;
  bool usedFallback = false;
  std::string diagnostic;

  bool succeeded() const { return backend != nullptr; }
};

TileOffloadBackendSelection selectTileOffloadBackend(
    const TileOffloadKernelPlan &plan,
    llvm::ArrayRef<const TileOffloadCodegenBackend *> availableBackends,
    llvm::StringRef preferredBackend, llvm::StringRef fallbackBackend,
    bool allowFallback);

} // namespace fir::TileOffload

#endif // FORTRAN_OPTIMIZER_DIALECT_TILEOFFLOAD_KERNEL_PLAN_H

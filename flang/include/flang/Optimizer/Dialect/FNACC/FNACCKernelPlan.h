#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELPLAN_H
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELPLAN_H

#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace fir::fnacc {

/// Compiler-side device IR produced by an FNACC code-generation backend.
/// These values describe intermediate artifacts, not necessarily something a
/// runtime loader can consume directly.
enum class FNACCDeviceIRKind { TTIR, LLVMIR, CUDATileIR, PTX };

/// Runtime-consumable CUDA device image formats.
enum class FNACCDeviceImageKind { PTX, Cubin };

inline llvm::StringRef fnaccDeviceIRKindName(FNACCDeviceIRKind kind) {
  switch (kind) {
  case FNACCDeviceIRKind::TTIR:
    return "ttir";
  case FNACCDeviceIRKind::LLVMIR:
    return "llvm-ir";
  case FNACCDeviceIRKind::CUDATileIR:
    return "cuda-tile-ir";
  case FNACCDeviceIRKind::PTX:
    return "ptx";
  }
  return "unknown";
}

inline llvm::StringRef fnaccDeviceImageKindName(FNACCDeviceImageKind kind) {
  switch (kind) {
  case FNACCDeviceImageKind::PTX:
    return "ptx";
  case FNACCDeviceImageKind::Cubin:
    return "cubin";
  }
  return "unknown";
}

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
  ArrayStride
};

enum class FNACCKernelParameterPassing { DevicePointer, Value };

/// One source-level parameter in the stable FNACC kernel ABI. Backend-private
/// parameters, such as parameters appended by Triton/NVVM, are deliberately
/// not represented here.
struct FNACCKernelParameter {
  unsigned slot = 0;
  FNACCKernelParameterRole role = FNACCKernelParameterRole::Read;
  FNACCKernelParameterPassing passing = FNACCKernelParameterPassing::Value;
  ElementType elementType = ElementType::Unknown;
  std::string name;
  int32_t arrayIndex = -1;
  int32_t scalarIndex = -1;
  int32_t dimension = -1;
};

struct FNACCPackBinding {
  unsigned kernelArgSlot = 0;
  int32_t target = 0;
};

struct FNACCKernelABI {
  llvm::SmallVector<FNACCKernelParameter> parameters;
  llvm::SmallVector<FNACCPackBinding> packBindings;
};

struct FNACCTileShape {
  int64_t x = 1;
  int64_t y = 1;
  int64_t z = 1;
};

enum class FNACCMatmulStrategy { Dot, Reduce, FMA };

/// Scheduling requests expressed without naming a particular backend. A
/// backend may reject a schedule or map it to its closest native concept.
struct FNACCKernelSchedule {
  FNACCTileShape tile;
  int32_t parallelSubgroups = 1;
  int32_t subgroupWidth = 32;
  int32_t pipelineStages = 3;
  FNACCMatmulStrategy f64MatmulStrategy = FNACCMatmulStrategy::Reduce;
};

struct FNACCReductionStagePlan {
  int32_t id = -1;
  std::string name;
  ReductionOperator reductionOperator = ReductionOperator::Add;
  ElementType elementType = ElementType::Unknown;
  FNACCKernelABI abi;
};

/// Backend-neutral description of one recognized FNACC launch.
///
/// ElementwiseKernel retains the recognized FIR values and expression tree.
/// The remaining fields contain stable identity, schedule and ABI information
/// that used to be reconstructed inside the Triton emitter.
struct FNACCKernelPlan {
  fir::fnacc::LaunchOp launchOp;
  int32_t id = -1;
  std::string name;
  ElementwiseKernel kernel;
  bool usesVariadicABI = false;
  bool copyBackWrites = true;
  FNACCKernelSchedule schedule;
  FNACCKernelABI abi;
  std::optional<FNACCReductionStagePlan> reductionStage;
};

struct FNACCKernelPlanOptions {
  int32_t requestedParallelSubgroups = 1;
  int32_t subgroupWidth = 32;
  int32_t pipelineStages = 3;
  FNACCMatmulStrategy f64MatmulStrategy = FNACCMatmulStrategy::Reduce;
};

class FNACCKernelPlanResult {
public:
  static FNACCKernelPlanResult success(FNACCKernelPlan plan);
  static FNACCKernelPlanResult failure(mlir::Operation *where,
                                       std::string reason);

  bool succeeded() const { return plan.has_value(); }
  bool failed() const { return !succeeded(); }

  const FNACCKernelPlan &getPlan() const { return *plan; }
  FNACCKernelPlan takePlan();
  const RecognitionFailure &getFailure() const { return failureInfo; }

private:
  std::optional<FNACCKernelPlan> plan;
  RecognitionFailure failureInfo;
};

FNACCKernelPlanResult
buildFNACCKernelPlan(fir::fnacc::LaunchOp launchOp, int32_t fallbackId,
                     int32_t nextSyntheticKernelId,
                     const FNACCKernelPlanOptions &options);

bool isReductionKernelKind(ElementwiseKernelKind kind);
llvm::StringRef fnaccKernelKindName(ElementwiseKernelKind kind);

struct FNACCBackendSupport {
  bool supported = false;
  std::string reason;

  static FNACCBackendSupport success() { return {true, {}}; }
  static FNACCBackendSupport failure(llvm::StringRef reason) {
    return {false, reason.str()};
  }
};

/// Interface shared by FNACC device-code backends. Module framing and kernel
/// emission use raw_ostream so textual and bytecode backends can share the
/// orchestration layer.
class FNACCCodegenBackend {
public:
  virtual ~FNACCCodegenBackend() = default;

  virtual llvm::StringRef getName() const = 0;
  virtual FNACCDeviceIRKind getDeviceIRKind() const {
    return FNACCDeviceIRKind::TTIR;
  }
  virtual FNACCDeviceImageKind getRuntimeImageKind() const = 0;
  virtual FNACCBackendSupport
  querySupport(const FNACCKernelPlan &plan) const = 0;

  virtual void beginModule(const FNACCKernelPlanOptions &options,
                           llvm::raw_ostream &os) const = 0;
  virtual mlir::LogicalResult emitKernel(const FNACCKernelPlan &plan,
                                         llvm::raw_ostream &os) const = 0;
  virtual void endModule(llvm::raw_ostream &os) const = 0;

  /// Number of backend-private pointer arguments appended after the stable
  /// FNACC ABI. This preserves compatibility with the current runtime while
  /// keeping those arguments out of FNACCKernelABI.
  virtual int32_t
  getPrivatePointerArgumentCount(const FNACCKernelPlan &plan) const = 0;
};

struct FNACCBackendSelection {
  const FNACCCodegenBackend *backend = nullptr;
  bool usedFallback = false;
  std::string diagnostic;

  bool succeeded() const { return backend != nullptr; }
};

FNACCBackendSelection selectFNACCBackend(
    const FNACCKernelPlan &plan,
    llvm::ArrayRef<const FNACCCodegenBackend *> availableBackends,
    llvm::StringRef preferredBackend, llvm::StringRef fallbackBackend,
    bool allowFallback);

} // namespace fir::fnacc

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELPLAN_H

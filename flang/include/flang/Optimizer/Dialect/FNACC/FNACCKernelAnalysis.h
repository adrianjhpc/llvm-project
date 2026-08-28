#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELANALYSIS_H
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELANALYSIS_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace fir::fnacc {

enum class ElementwiseExtentSourceKind {
  Unknown,
  ConstantInteger,
  Value,
  LoadIntegerRef,
  BoxDim,
  BoxLowerBound
};

struct ElementwiseExtentSource {
  ElementwiseExtentSourceKind kind = ElementwiseExtentSourceKind::Unknown;

  // Meaning depends on kind:
  //
  // Value:
  //   An SSA value usable directly as an extent.
  //
  // LoadIntegerRef:
  //   An !fir.ref<iN> or equivalent scalar reference. Runtime lowering will
  //   emit fir.load outside the launch and convert the extent to i32.
  //
  // BoxDim:
  //   A !fir.box<!fir.array<...>>. Runtime lowering will emit fir.box_dims
  //   outside the launch and use result #1, the extent.
  mlir::Value value;

  // For ConstantInteger.
  int64_t constantValue = 0;

  // For BoxDim. Zero-based dimension number.
  unsigned dim = 0;
};

enum class ElementwiseKernelKind {
  BinaryArrayArray,
  Saxpy1D,
  Expr1D,
  MultiExpr1D,
  Expr2D,
  Stencil2D,
  MatMul2D,
  ReductionSum1D,
  ReductionDot1D,
  ReductionProduct1D,
  ReductionMin1D,
  ReductionMax1D
};

/// Return true when the kernel uses the staged, dynamically-sized launch ABI.
bool usesVariadicLaunchABI(ElementwiseKernelKind kind);

/// Return true for a primary (non-synthetic) reduction kernel kind.
bool isReductionKernelKind(ElementwiseKernelKind kind);

/// Stable values written to fnacc.reduction_ops and kernel JSON.
enum class ReductionOperator : int32_t {
  Add = 0,
  Multiply = 1,
  Min = 2,
  Max = 3
};

enum class ElementwiseExprKind {
  ArrayLoad,
  ScalarLoad,
  ConstantReal,
  ConstantInteger,

  // Unary arithmetic.
  NegF,
  AbsF,
  SqrtF,
  ExpF,
  LogF,
  SinF,
  CosF,
  TanhF,
  SquareF,
  AbsI,

  // Binary arithmetic.
  AddF,
  SubF,
  MulF,
  DivF,
  MinF,
  MaxF,
  MinNumF,
  MaxNumF,
  AddI,
  SubI,
  MulI,
  DivSI,
  MinSI,
  MaxSI,

  // Comparisons.
  CmpOLT,
  CmpOLE,
  CmpOGT,
  CmpOGE,
  CmpOEQ,
  CmpONE,
  CmpSLT,
  CmpSLE,
  CmpSGT,
  CmpSGE,
  CmpIEQ,
  CmpINE,

  // Predicate composition.
  And,
  Or,

  // Conditional value selection.
  Select
};

enum class ElementwiseExprResultKind { Element, Predicate };

struct ElementwiseExpr {
  ElementwiseExprKind kind;
  ElementwiseExprResultKind resultKind = ElementwiseExprResultKind::Element;

  mlir::Value source;
  // Index in ElementwiseKernel::arrayAccesses. This distinguishes two loads
  // from the same array at different stencil offsets.
  int32_t arrayAccessIndex = -1;
  double realValue = 0.0;
  int64_t integerValue = 0;

  llvm::SmallVector<std::unique_ptr<ElementwiseExpr>> operands;
};

enum class ElementType { Unknown, I8, I16, I32, I64, F32, F64 };

enum class ScalarCaptureKind { Reference, Value };

struct ScalarCapture {
  ScalarCaptureKind kind;
  mlir::Value value;
};

struct ElementwiseArrayAccess {
  mlir::Value array;
  mlir::Value loadedValue;
  unsigned arrayArgumentIndex = 0;
  /// For each array dimension, the logical kernel dimension that supplies its
  /// subscript. A rank-1 coordinate array in a rank-2 stencil therefore uses
  /// either {0} (the inner/X loop) or {1} (the outer/Y loop).
  llvm::SmallVector<unsigned, 3> dimensions;
  llvm::SmallVector<int64_t, 3> offsets;
};

struct ElementwiseOutput {
  mlir::Value array;
  mlir::Value storedValue;
  unsigned arrayArgumentIndex = 0;
  llvm::SmallVector<unsigned, 3> dimensions;
  llvm::SmallVector<int64_t, 3> offsets;
  std::unique_ptr<ElementwiseExpr> expression;
};

struct ElementwiseArrayArgument {
  mlir::Value array;
  bool read = false;
  bool write = false;
  /// Physical rank of this array binding. This is independent of the kernel
  /// iteration rank for mixed-rank stencils.
  unsigned rank = 0;
};

/// Classification of scalar storage referenced by a parallel loop.
///
/// ReadOnlyCapture is materialized outside fnacc.launch and passed by value.
/// IterationPrivate is assigned and consumed within one logical loop
/// iteration; its defining expression is promoted to device SSA. A remaining
/// mutable reference is unsafe because it would otherwise become one uniform
/// kernel argument shared by all logical iterations.
enum class ScalarReferenceKind {
  ReadOnlyCapture,
  IterationPrivate,
  UnsupportedMutable
};

struct ElementwiseKernel {
  int32_t rank = 1;

  ElementwiseKernelKind kind = ElementwiseKernelKind::BinaryArrayArray;

  fir::DoLoopOp loop1D;

  fir::DoLoopOp outerLoop;
  fir::DoLoopOp innerLoop;

  // For matrix multiplication:
  //
  //   do j = ...
  //     do i = ...
  //       acc = 0
  //       do p = ...
  //         acc = acc + a(i,p) * b(p,j)
  //       end do
  //       c(i,j) = acc
  //     end do
  //   end do
  //
  // outerLoop    = j loop
  // innerLoop    = i loop
  // reductionLoop = p loop
  fir::DoLoopOp reductionLoop;

  // Runtime extent sources.
  //
  // For 1-D:
  //   extentX = loop trip upper extent.
  //
  // For 2-D:
  //   extentX = inner/i extent.
  //   extentY = outer/j extent.
  ElementwiseExtentSource extentX;
  ElementwiseExtentSource extentY;
  ElementwiseExtentSource extentZ;
  ElementwiseExtentSource loopLowerX;
  ElementwiseExtentSource loopLowerY;
  ElementwiseExtentSource loopLowerZ;

  mlir::Value innerIndMemref;
  mlir::Value outerIndMemref;

  mlir::Value reductionIndMemref;
  mlir::Value accumulatorMemref;
  mlir::Value reductionScalarRef;

  llvm::SmallVector<mlir::Value> readArrays;
  mlir::Value writeArray;

  /// Unique array bindings for the variadic ABI. Stencil accesses retain their
  /// constant affine offsets, and each stencil output owns an expression root.
  llvm::SmallVector<ElementwiseArrayArgument> arrayArguments;
  llvm::SmallVector<ElementwiseArrayAccess> arrayAccesses;
  llvm::SmallVector<ElementwiseOutput> outputs;
  llvm::SmallVector<mlir::Value> writeArrays;

  /// Host-visible, read-only scalar references passed to the kernel by value.
  llvm::SmallVector<mlir::Value> scalarRefs;

  /// Scalar references proven iteration-private and promoted into the
  /// expression tree. They are recorded for diagnostics and validation only;
  /// they are deliberately absent from the runtime ABI.
  llvm::SmallVector<mlir::Value> privateScalarRefs;

  //  llvm::SmallVector<ScalarCapture> scalarCaptures;

  mlir::Operation *computeOp = nullptr;

  ReductionOperator reductionOperator = ReductionOperator::Add;

  std::unique_ptr<ElementwiseExpr> expression;

  ElementType elementType = ElementType::Unknown;

  /// Operations proven to belong to the recognized kernel. Runtime lowering
  /// must not erase the launch unless every operation in the launch region is
  /// either present here or is an explicitly accepted structural operation.
  llvm::SmallVector<mlir::Operation *, 64> consumedOps;
};

struct RecognitionFailure {
  mlir::Operation *where = nullptr;
  std::string reason;
};

class ElementwiseRecognitionResult {
public:
  static ElementwiseRecognitionResult success(ElementwiseKernel kernel) {
    ElementwiseRecognitionResult result;
    result.kernel = std::move(kernel);
    return result;
  }

  static ElementwiseRecognitionResult failure(mlir::Operation *where,
                                              std::string reason) {
    ElementwiseRecognitionResult result;
    result.failureInfo.where = where;
    result.failureInfo.reason = std::move(reason);
    return result;
  }

  bool succeeded() const { return kernel.has_value(); }
  bool failed() const { return !succeeded(); }

  ElementwiseKernel &getKernel() { return *kernel; }
  const ElementwiseKernel &getKernel() const { return *kernel; }

  const RecognitionFailure &getFailure() const { return failureInfo; }

private:
  std::optional<ElementwiseKernel> kernel;
  RecognitionFailure failureInfo;
};

ElementwiseRecognitionResult
recognizeElementwiseKernel(fir::fnacc::LaunchOp launchOp);

bool isSupportedElementwiseCompute(mlir::Operation *op);

} // namespace fir::fnacc

#endif // FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELANALYSIS_H

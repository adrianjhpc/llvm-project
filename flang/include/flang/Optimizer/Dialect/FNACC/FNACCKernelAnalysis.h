#ifndef FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELANALYSIS_H
#define FORTRAN_OPTIMIZER_DIALECT_FNACC_FNACCKERNELANALYSIS_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <string>

namespace fir::fnacc {

enum class ElementwiseExtentSourceKind {
  Unknown,
  Value,
  LoadIntegerRef,
  BoxDim
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

  // For BoxDim. Zero-based dimension number.
  unsigned dim = 0;
};

enum class ElementwiseKernelKind {
  BinaryArrayArray,
  Saxpy1D,
  Expr1D,
  Expr2D,
  MatMul2D,
  ReductionSum1D,
  ReductionDot1D,
  ReductionProduct1D,
  ReductionMin1D,
  ReductionMax1D
};

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
  AddF,
  SubF,
  MulF,
  DivF
};

struct ElementwiseExpr {
  ElementwiseExprKind kind;

  // For ArrayLoad and ScalarLoad.
  mlir::Value source;

  // For ConstantReal.
  double realValue = 0.0;

  llvm::SmallVector<std::unique_ptr<ElementwiseExpr>> operands;
};

enum class ElementType { Unknown, F32, F64 };

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

  mlir::Value innerIndMemref;
  mlir::Value outerIndMemref;

  mlir::Value reductionIndMemref;
  mlir::Value accumulatorMemref;
  mlir::Value reductionScalarRef;

  llvm::SmallVector<mlir::Value> readArrays;
  mlir::Value writeArray;

  llvm::SmallVector<mlir::Value> scalarRefs;

  mlir::Operation *computeOp = nullptr;

  ReductionOperator reductionOperator = ReductionOperator::Add;

  std::unique_ptr<ElementwiseExpr> expression;

  ElementType elementType = ElementType::Unknown;
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

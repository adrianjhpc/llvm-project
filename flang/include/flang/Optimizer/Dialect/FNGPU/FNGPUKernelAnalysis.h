#ifndef FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUKERNELANALYSIS_H
#define FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUKERNELANALYSIS_H

#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <memory>

namespace fir::fngpu {

  enum class ElementwiseKernelKind {
    BinaryArrayArray,
    Saxpy1D,
    Expr1D
  };

  enum class ElementwiseExprKind {
    ArrayLoad,
    ScalarLoad,
    ConstantF32,
    AddF,
    SubF,
    MulF,
    DivF
  };

  struct ElementwiseExpr {
    ElementwiseExprKind kind;

    // For ArrayLoad and ScalarLoad.
    mlir::Value source;

    // For ConstantF32.
    double f32Value = 0.0;

    llvm::SmallVector<std::unique_ptr<ElementwiseExpr>> operands;
  };

  struct ElementwiseKernel {
    int32_t rank = 1;

    ElementwiseKernelKind kind = ElementwiseKernelKind::BinaryArrayArray;

    // 1-D case.
    fir::DoLoopOp loop1D;

    // 2-D case.
    // outerLoop = j loop
    // innerLoop = i loop
    fir::DoLoopOp outerLoop;
    fir::DoLoopOp innerLoop;

    // Runtime extent references.
    // For 1-D:
    //   nRef = n
    //
    // For 2-D:
    //   nRef = x extent / first dimension / i bound
    //   mRef = y extent / second dimension / j bound
    mlir::Value nRef;
    mlir::Value mRef;

    // Induction-variable storage discovered from:
    //
    //   fir.store %iv to %i
    mlir::Value innerIndMemref;
    mlir::Value outerIndMemref;

    llvm::SmallVector<mlir::Value> readArrays;
    mlir::Value writeArray;

    llvm::SmallVector<mlir::Value> scalarRefs;
    
    mlir::Operation *computeOp = nullptr;

    // Generic 1-D expression tree.
    //
    // Used for kernels such as:
    //
    //   c(i) = alpha * a(i) + beta * b(i)
    std::unique_ptr<ElementwiseExpr> expression;

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
  recognizeElementwiseKernel(fir::fngpu::LaunchOp launchOp);

  bool isSupportedElementwiseCompute(mlir::Operation *op);

} // namespace fir::fngpu

#endif // FORTRAN_OPTIMIZER_DIALECT_FNGPU_FNGPUKERNELANALYSIS_H


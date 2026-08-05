#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace fir::fngpu;

// The .inc files automatically generate the fir::fngpu namespaces
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.cpp.inc"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FNGPU/FNGPUOps.cpp.inc"

// Explicitly scope the initialize method to the namespace
void fir::fngpu::FNGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/FNGPU/FNGPUOps.cpp.inc"
      >();
}

llvm::LogicalResult fir::fngpu::LaunchOp::verify() {
  if (getPackVars().size() != getPackTargets().size())
    return emitOpError(
        "expected pack_targets to have exactly one entry per pack var");

  for (int32_t t : getPackTargets())
    if (t != 0 && t != 1)
      return emitOpError("pack target must be 0 (host) or 1 (device)");
  return mlir::success();
}

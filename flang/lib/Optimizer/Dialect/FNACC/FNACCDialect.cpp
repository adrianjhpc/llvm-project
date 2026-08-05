#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace fir::fnacc;

// The .inc files automatically generate the fir::fnacc namespaces
#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.cpp.inc"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FNACC/FNACCOps.cpp.inc"

// Explicitly scope the initialize method to the namespace
void fir::fnacc::FNACCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/FNACC/FNACCOps.cpp.inc"
      >();
}

llvm::LogicalResult fir::fnacc::LaunchOp::verify() {
  if (getPackVars().size() != getPackTargets().size())
    return emitOpError(
        "expected pack_targets to have exactly one entry per pack var");

  for (int32_t t : getPackTargets())
    if (t != 0 && t != 1)
      return emitOpError("pack target must be 0 (host) or 1 (device)");
  return mlir::success();
}

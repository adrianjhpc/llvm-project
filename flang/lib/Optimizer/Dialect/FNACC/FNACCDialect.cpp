#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseSet.h"

#include <cstddef>
#include <limits>

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
  llvm::ArrayRef<int64_t> tileSizes = getTileSizes();

  if (tileSizes.size() > 3)
    return emitOpError("expected at most three tile sizes");

  int64_t tileProduct = 1;
  for (int64_t tile : tileSizes) {
    if (tile <= 0)
      return emitOpError("tile sizes must be positive");
    if (tile > std::numeric_limits<int32_t>::max())
      return emitOpError("tile size is outside the runtime i32 range");
    if (tileProduct > std::numeric_limits<int32_t>::max() / tile)
      return emitOpError("tile-size product exceeds the runtime i32 range");
    tileProduct *= tile;
  }

  if (getPackVars().size() != getPackTargets().size())
    return emitOpError(
        "expected pack_targets to have exactly one entry per pack var");

  llvm::DenseSet<mlir::Value> seenPackVars;
  for (mlir::Value var : getPackVars()) {
    if (!seenPackVars.insert(var).second)
      return emitOpError(
          "the same variable appears more than once in PACK/REDUCTION");
  }

  for (int32_t t : getPackTargets()) {
    if (t != 0 && t != 1)
      return emitOpError("pack target must be 0 (host) or 1 (device)");
  }

  auto reductionSlots =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>("fnacc.reduction_slots");
  auto reductionOps =
      (*this)->getAttrOfType<mlir::DenseI32ArrayAttr>("fnacc.reduction_ops");

  if (static_cast<bool>(reductionSlots) != static_cast<bool>(reductionOps))
    return emitOpError(
        "expected reduction_slots and reduction_ops to appear together");

  if (reductionSlots) {
    if (reductionSlots.size() != reductionOps.size())
      return emitOpError("expected one reduction operation per reduction slot");

    llvm::DenseSet<int32_t> seenSlots;
    auto slots = reductionSlots.asArrayRef();
    auto ops = reductionOps.asArrayRef();

    for (std::size_t i = 0; i < slots.size(); ++i) {
      int32_t slot = slots[i];
      int32_t op = ops[i];
      if (slot < 0 || static_cast<unsigned>(slot) >= getPackVars().size())
        return emitOpError("reduction slot is outside the pack variable list");
      if (!seenSlots.insert(slot).second)
        return emitOpError("duplicate reduction slot");
      if (op != 0)
        return emitOpError("only SUM reduction operation code 0 is supported");
    }
  }

  return mlir::success();
}

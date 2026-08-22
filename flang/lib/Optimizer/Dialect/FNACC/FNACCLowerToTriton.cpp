#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <cstdint>
#include <limits>

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCLOWERTOTRITON
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

static constexpr int32_t kTritonHiddenPtrArgs = 2;

static constexpr llvm::StringLiteral kKernelIdAttrName = "fnacc.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fnacc.kernel_name";

static StringRef ttElementType(fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::F32:
    return "f32";
  case fir::fnacc::ElementType::F64:
    return "f64";
  default:
    llvm_unreachable("unsupported FNACC element type");
  }
}

static std::string ptrType(fir::fnacc::ElementType type) {
  return std::string("!tt.ptr<") + ttElementType(type).str() + ">";
}

static std::string tensorType(int64_t n, fir::fnacc::ElementType type) {
  return std::string("tensor<") + std::to_string(n) + "x" +
         ttElementType(type).str() + ">";
}

static std::string ptrTensorType(int64_t n, fir::fnacc::ElementType type) {
  return std::string("tensor<") + std::to_string(n) + "x" + ptrType(type) + ">";
}

static std::string tensor2DType(int64_t x, int64_t y,
                                fir::fnacc::ElementType type) {
  return std::string("tensor<") + std::to_string(x) + "x" + std::to_string(y) +
         "x" + ttElementType(type).str() + ">";
}

static std::string ptrTensor2DType(int64_t x, int64_t y,
                                   fir::fnacc::ElementType type) {
  return std::string("tensor<") + std::to_string(x) + "x" + std::to_string(y) +
         "x" + ptrType(type) + ">";
}

static std::string jsonElementType(fir::fnacc::ElementType type) {
  return ttElementType(type).str();
}

static std::string jsonPtrType(fir::fnacc::ElementType type) {
  return std::string("ptr<") + jsonElementType(type) + ">";
}

static int32_t getKernelId(fir::fnacc::LaunchOp launchOp, int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName))
    return static_cast<int32_t>(attr.getInt());
  return fallbackId;
}

static std::string getKernelName(fir::fnacc::LaunchOp launchOp,
                                 int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<StringAttr>(kKernelNameAttrName))
    return attr.getValue().str();
  return "fnacc_kernel_" + std::to_string(fallbackId);
}

static int32_t getKernelNumWarps(const fir::fnacc::ElementwiseKernel &k,
                                 int32_t requestedNumWarps) {
  using Kind = fir::fnacc::ElementwiseKernelKind;
  using Elem = fir::fnacc::ElementType;

  // f64 scalar elementwise kernels such as:
  //
  //   c(i) = alpha * a(i) + b(i)
  //   c(i) = alpha * a(i) + beta * b(i)
  //
  // are more compute-heavy than plain vector add. Let them use the requested
  // number of warps.
  bool isF64ScalarElementwise =
      k.elementType == Elem::F64 && !k.scalarRefs.empty() &&
      (k.kind == Kind::Saxpy1D || k.kind == Kind::Expr1D ||
       k.kind == Kind::Expr2D);

  if (isF64ScalarElementwise)
    return requestedNumWarps;

  switch (k.kind) {
  case Kind::ReductionSum1D:
  case Kind::ReductionDot1D:
  case Kind::ReductionProduct1D:
  case Kind::ReductionMin1D:
  case Kind::ReductionMax1D:
    // The pass default remains one warp. Values greater than one are recorded
    // for both the primary and hierarchical follow-up kernels; the driver
    // completes lowering of any residual warp-id query automatically.
    return requestedNumWarps;

  case Kind::BinaryArrayArray:
  case Kind::Saxpy1D:
  case Kind::Expr1D:
  case Kind::Expr2D:
    return 1;

  case Kind::MatMul2D:
    return requestedNumWarps;
  }

  llvm_unreachable("unknown FNACC kernel kind");
}

static StringRef ttArith(Operation *op) {
  if (isa<arith::AddFOp>(op))
    return "arith.addf";
  if (isa<arith::SubFOp>(op))
    return "arith.subf";
  if (isa<arith::MulFOp>(op))
    return "arith.mulf";
  return "arith.divf";
}

static StringRef ttArithForExprKind(fir::fnacc::ElementwiseExprKind kind) {
  switch (kind) {
  case fir::fnacc::ElementwiseExprKind::AddF:
    return "arith.addf";
  case fir::fnacc::ElementwiseExprKind::SubF:
    return "arith.subf";
  case fir::fnacc::ElementwiseExprKind::MulF:
    return "arith.mulf";
  case fir::fnacc::ElementwiseExprKind::DivF:
    return "arith.divf";
  default:
    llvm_unreachable("not a binary arithmetic expression kind");
  }
}

static int findValueIndex(ArrayRef<Value> values, Value value) {
  for (auto it : llvm::enumerate(values))
    if (it.value() == value)
      return static_cast<int>(it.index());
  return -1;
}

static void emitLoad1D(StringRef ptr, StringRef dst, int64_t block,
                       fir::fnacc::ElementType type, llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(type);
  std::string ptrVecTy = ptrTensorType(block, type);

  os << "  " << dst << "p = tt.splat " << ptr << " : " << ptrTy << " -> "
     << ptrVecTy << "\n";

  os << "  " << dst << "o = tt.addptr " << dst << "p, %offs : " << ptrVecTy
     << ", tensor<" << block << "xi32>\n";

  os << "  " << dst << " = tt.load " << dst << "o, %mask : " << ptrVecTy
     << "\n";
}

static void emitStore1D(StringRef ptr, StringRef value, int64_t block,
                        fir::fnacc::ElementType type, llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(type);
  std::string ptrVecTy = ptrTensorType(block, type);

  os << "  %cp = tt.splat " << ptr << " : " << ptrTy << " -> " << ptrVecTy
     << "\n";

  os << "  %co = tt.addptr %cp, %offs : " << ptrVecTy << ", tensor<" << block
     << "xi32>\n";

  os << "  tt.store %co, " << value << ", %mask : " << ptrVecTy << "\n";
}

static void emitTriton1D(const fir::fnacc::ElementwiseKernel &k, int64_t block,
                         StringRef kernelName, llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  assert(k.readArrays.size() >= 2 &&
         "binary 1-D kernel requires two read arrays");

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";
  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %bS   = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %bS, %rng : tensor<" << block << "xi32>\n";
  os << "  %nS   = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %nS : tensor<" << block << "xi32>\n";

  emitLoad1D("%a", "%av", block, k.elementType, os);
  emitLoad1D("%b", "%bv", block, k.elementType, os);

  os << "  %r = " << ttArith(k.computeOp) << " %av, %bv : tensor<" << block
     << "x" << elemTy << ">\n";

  emitStore1D("%c", "%r", block, k.elementType, os);

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonSaxpy1D(const fir::fnacc::ElementwiseKernel &k,
                              int64_t block, StringRef kernelName,
                              llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy << ", %alpha: " << elemTy
     << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";
  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %bS   = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %bS, %rng : tensor<" << block << "xi32>\n";
  os << "  %nS   = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %nS : tensor<" << block << "xi32>\n";

  emitLoad1D("%a", "%av", block, k.elementType, os);
  emitLoad1D("%b", "%bv", block, k.elementType, os);

  os << "  %alpha_s = tt.splat %alpha : " << elemTy << " -> tensor<" << block
     << "x" << elemTy << ">\n";

  os << "  %scaled = arith.mulf %alpha_s, %av : tensor<" << block << "x"
     << elemTy << ">\n";
  os << "  %r = arith.addf %scaled, %bv : tensor<" << block << "x" << elemTy
     << ">\n";

  emitStore1D("%c", "%r", block, k.elementType, os);

  os << "  tt.return\n";
  os << "}\n\n";
}

struct ExprTritonEmitterState {
  int64_t block = 0;
  unsigned nextTmp = 0;
  llvm::SmallVector<bool> scalarSplatEmitted;
  llvm::SmallVector<std::string> scalarSplatNames;
};

static std::string emitExprVector(const fir::fnacc::ElementwiseKernel &k,
                                  const fir::fnacc::ElementwiseExpr &expr,
                                  ExprTritonEmitterState &state,
                                  llvm::raw_ostream &os) {
  int64_t block = state.block;
  std::string elemTy = ttElementType(k.elementType).str();

  switch (expr.kind) {
  case fir::fnacc::ElementwiseExprKind::ArrayLoad: {
    int index = findValueIndex(k.readArrays, expr.source);
    assert(index >= 0 && "array load source not found in read array list");

    if (index == 0)
      return "%read0v";
    if (index == 1)
      return "%read1v";
    if (index == 2)
      return "%read2v";

    llvm_unreachable("only three read arrays are supported");
  }

  case fir::fnacc::ElementwiseExprKind::ScalarLoad: {
    int index = findValueIndex(k.scalarRefs, expr.source);
    assert(index >= 0 && "scalar source not found in scalar list");

    if (static_cast<unsigned>(index) >= state.scalarSplatEmitted.size()) {
      state.scalarSplatEmitted.resize(index + 1, false);
      state.scalarSplatNames.resize(index + 1);
    }

    if (!state.scalarSplatEmitted[index]) {
      std::string scalarName = "%scalar" + std::to_string(index);
      std::string splatName = "%scalar" + std::to_string(index) + "_s";

      os << "  " << splatName << " = tt.splat " << scalarName << " : " << elemTy
         << " -> tensor<" << block << "x" << elemTy << ">\n";

      state.scalarSplatEmitted[index] = true;
      state.scalarSplatNames[index] = splatName;
    }

    return state.scalarSplatNames[index];
  }

  case fir::fnacc::ElementwiseExprKind::ConstantReal: {
    unsigned id = state.nextTmp++;
    std::string cst = "%cst" + std::to_string(id);
    std::string splat = "%cst" + std::to_string(id) + "_s";

    os << "  " << cst << " = arith.constant " << expr.realValue << " : "
       << elemTy << "\n";

    os << "  " << splat << " = tt.splat " << cst << " : " << elemTy
       << " -> tensor<" << block << "x" << elemTy << ">\n";

    return splat;
  }

  case fir::fnacc::ElementwiseExprKind::AddF:
  case fir::fnacc::ElementwiseExprKind::SubF:
  case fir::fnacc::ElementwiseExprKind::MulF:
  case fir::fnacc::ElementwiseExprKind::DivF: {
    assert(expr.operands.size() == 2 && "binary expression expected");

    std::string lhs = emitExprVector(k, *expr.operands[0], state, os);
    std::string rhs = emitExprVector(k, *expr.operands[1], state, os);

    std::string result = "%expr" + std::to_string(state.nextTmp++);

    StringRef opName = ttArithForExprKind(expr.kind);

    os << "  " << result << " = " << opName << " " << lhs << ", " << rhs;
    os << " : tensor<" << block << "x" << elemTy << ">\n";

    return result;
  }
  }

  llvm_unreachable("unknown FNACC expression kind");
}

static void emitTritonExpr1D(const fir::fnacc::ElementwiseKernel &k,
                             int64_t block, StringRef kernelName,
                             llvm::raw_ostream &os) {
  assert(k.expression && "Expr1D kernel has no expression tree");
  assert(!k.readArrays.empty() && k.readArrays.size() <= 3 &&
         "Expr1D requires one to three read arrays");

  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(";

  for (unsigned i = 0; i < k.readArrays.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << "%read" << i << ": " << ptrTy;
  }

  os << ", %c: " << ptrTy;

  for (unsigned i = 0; i < k.scalarRefs.size(); ++i)
    os << ", %scalar" << i << ": " << elemTy;

  os << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";
  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %bS   = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %bS, %rng : tensor<" << block << "xi32>\n";
  os << "  %nS   = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %nS : tensor<" << block << "xi32>\n";

  if (k.readArrays.size() >= 1)
    emitLoad1D("%read0", "%read0v", block, k.elementType, os);
  if (k.readArrays.size() >= 2)
    emitLoad1D("%read1", "%read1v", block, k.elementType, os);
  if (k.readArrays.size() >= 3)
    emitLoad1D("%read2", "%read2v", block, k.elementType, os);

  ExprTritonEmitterState state;
  state.block = block;
  state.scalarSplatEmitted.resize(k.scalarRefs.size(), false);
  state.scalarSplatNames.resize(k.scalarRefs.size());

  std::string result = emitExprVector(k, *k.expression, state, os);

  emitStore1D("%c", result, block, k.elementType, os);

  os << "  tt.return\n";
  os << "}\n\n";
}

static StringRef reductionOperatorName(fir::fnacc::ReductionOperator op) {
  switch (op) {
  case fir::fnacc::ReductionOperator::Add:
    return "add";
  case fir::fnacc::ReductionOperator::Multiply:
    return "multiply";
  case fir::fnacc::ReductionOperator::Min:
    return "min";
  case fir::fnacc::ReductionOperator::Max:
    return "max";
  }
  llvm_unreachable("unknown FNACC reduction operator");
}

static StringRef reductionArithOp(fir::fnacc::ReductionOperator op) {
  switch (op) {
  case fir::fnacc::ReductionOperator::Add:
    return "arith.addf";
  case fir::fnacc::ReductionOperator::Multiply:
    return "arith.mulf";
  case fir::fnacc::ReductionOperator::Min:
    return "arith.minimumf";
  case fir::fnacc::ReductionOperator::Max:
    return "arith.maximumf";
  }
  llvm_unreachable("unknown FNACC reduction operator");
}

static StringRef reductionIdentity(fir::fnacc::ReductionOperator op,
                                   fir::fnacc::ElementType type) {
  switch (op) {
  case fir::fnacc::ReductionOperator::Add:
    return "0.000000e+00";
  case fir::fnacc::ReductionOperator::Multiply:
    return "1.000000e+00";
  case fir::fnacc::ReductionOperator::Min:
    return type == fir::fnacc::ElementType::F64 ? "0x7FF0000000000000"
                                                : "0x7F800000";
  case fir::fnacc::ReductionOperator::Max:
    return type == fir::fnacc::ElementType::F64 ? "0xFFF0000000000000"
                                                : "0xFF800000";
  }
  llvm_unreachable("unknown FNACC reduction operator");
}

static void emitTritonReduction1D(const fir::fnacc::ElementwiseKernel &k,
                                  int64_t block, StringRef kernelName,
                                  llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(%a: " << ptrTy
     << ", %partials: " << ptrTy
     << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";
  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %base_s = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %base_s, %rng : tensor<" << block << "xi32>\n";
  os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %n_s : tensor<" << block << "xi32>\n";

  os << "  %ap = tt.splat %a : " << ptrTy << " -> tensor<" << block << "x"
     << ptrTy << ">\n";
  os << "  %ao = tt.addptr %ap, %offs : tensor<" << block << "x" << ptrTy
     << ">, tensor<" << block << "xi32>\n";
  os << "  %vals = tt.load %ao, %mask : tensor<" << block << "x" << ptrTy
     << ">\n";

  os << "  %identity = arith.constant "
     << reductionIdentity(k.reductionOperator, k.elementType) << " : " << elemTy
     << "\n";
  os << "  %identity_s = tt.splat %identity : " << elemTy << " -> tensor<"
     << block << "x" << elemTy << ">\n";
  os << "  %safe = arith.select %mask, %vals, %identity_s : tensor<" << block
     << "xi1>, tensor<" << block << "x" << elemTy << ">\n";

  os << "  %reduced = \"tt.reduce\"(%safe) ({\n";
  os << "  ^bb0(%lhs: " << elemTy << ", %rhs: " << elemTy << "):\n";
  os << "    %r = " << reductionArithOp(k.reductionOperator)
     << " %lhs, %rhs : " << elemTy << "\n";
  os << "    \"tt.reduce.return\"(%r) : (" << elemTy << ") -> ()\n";
  os << "  }) {axis = 0 : i32} : (tensor<" << block << "x" << elemTy << ">) -> "
     << elemTy << "\n";

  os << "  %outp = tt.addptr %partials, %pid : " << ptrTy << ", i32\n";
  os << "  tt.store %outp, %reduced : " << ptrTy << "\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

/// Emit the generic follow-up kernel used by hierarchical reductions.
/// Each program reduces one contiguous block of the previous stage's partial
/// sums and writes one value for the next stage. The runtime repeatedly invokes
/// this kernel until only a single device value remains.
static void emitTritonReductionStage1D(fir::fnacc::ElementType type,
                                       fir::fnacc::ReductionOperator op,
                                       int64_t block, StringRef kernelName,
                                       llvm::raw_ostream &os) {
  std::string ptrTy = ptrType(type);
  std::string elemTy = ttElementType(type).str();

  os << "tt.func @" << kernelName << "(%input: " << ptrTy
     << ", %output: " << ptrTy
     << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";
  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %base_s = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %base_s, %rng : tensor<" << block << "xi32>\n";
  os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %n_s : tensor<" << block << "xi32>\n";

  os << "  %input_s = tt.splat %input : " << ptrTy << " -> tensor<" << block
     << "x" << ptrTy << ">\n";
  os << "  %input_ptrs = tt.addptr %input_s, %offs : tensor<" << block << "x"
     << ptrTy << ">, tensor<" << block << "xi32>\n";
  os << "  %vals = tt.load %input_ptrs, %mask : tensor<" << block << "x"
     << ptrTy << ">\n";

  os << "  %identity = arith.constant " << reductionIdentity(op, type) << " : "
     << elemTy << "\n";
  os << "  %identity_s = tt.splat %identity : " << elemTy << " -> tensor<"
     << block << "x" << elemTy << ">\n";
  os << "  %safe = arith.select %mask, %vals, %identity_s : tensor<" << block
     << "xi1>, tensor<" << block << "x" << elemTy << ">\n";

  os << "  %reduced = \"tt.reduce\"(%safe) ({\n";
  os << "  ^bb0(%lhs: " << elemTy << ", %rhs: " << elemTy << "):\n";
  os << "    %r = " << reductionArithOp(op) << " %lhs, %rhs : " << elemTy
     << "\n";
  os << "    \"tt.reduce.return\"(%r) : (" << elemTy << ") -> ()\n";
  os << "  }) {axis = 0 : i32} : (tensor<" << block << "x" << elemTy << ">) -> "
     << elemTy << "\n";

  os << "  %outp = tt.addptr %output, %pid : " << ptrTy << ", i32\n";
  os << "  tt.store %outp, %reduced : " << ptrTy << "\n";
  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTriton2D(const fir::fnacc::ElementwiseKernel &k, int64_t blockX,
                         int64_t blockY, StringRef kernelName,
                         llvm::raw_ostream &os) {
  int64_t block = blockX * blockY;
  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();
  std::string ptrVecTy = ptrTensorType(block, k.elementType);

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy
     << ", %n: i32, %m: i32) attributes {noinline = false} {\n";

  os << "  %pid_x = tt.get_program_id x : i32\n";
  os << "  %pid_y = tt.get_program_id y : i32\n";
  os << "  %bx = arith.constant " << blockX << " : i32\n";
  os << "  %by = arith.constant " << blockY << " : i32\n";
  os << "  %base_x = arith.muli %pid_x, %bx : i32\n";
  os << "  %base_y = arith.muli %pid_y, %by : i32\n";
  os << "  %r = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %bx_s = tt.splat %bx : i32 -> tensor<" << block << "xi32>\n";
  os << "  %local_i = arith.remui %r, %bx_s : tensor<" << block << "xi32>\n";
  os << "  %local_j = arith.divui %r, %bx_s : tensor<" << block << "xi32>\n";
  os << "  %base_x_s = tt.splat %base_x : i32 -> tensor<" << block << "xi32>\n";
  os << "  %base_y_s = tt.splat %base_y : i32 -> tensor<" << block << "xi32>\n";
  os << "  %ix = arith.addi %base_x_s, %local_i : tensor<" << block
     << "xi32>\n";
  os << "  %jy = arith.addi %base_y_s, %local_j : tensor<" << block
     << "xi32>\n";
  os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %m_s = tt.splat %m : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask_x = arith.cmpi slt, %ix, %n_s : tensor<" << block << "xi32>\n";
  os << "  %mask_y = arith.cmpi slt, %jy, %m_s : tensor<" << block << "xi32>\n";
  os << "  %mask = arith.andi %mask_x, %mask_y : tensor<" << block << "xi1>\n";
  os << "  %jy_n = arith.muli %jy, %n_s : tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %ix, %jy_n : tensor<" << block << "xi32>\n";

  emitLoad1D("%a", "%av", block, k.elementType, os);
  emitLoad1D("%b", "%bv", block, k.elementType, os);

  os << "  %rval = " << ttArith(k.computeOp) << " %av, %bv : tensor<" << block
     << "x" << elemTy << ">\n";

  os << "  %cp = tt.splat %c : " << ptrTy << " -> " << ptrVecTy << "\n";
  os << "  %co = tt.addptr %cp, %offs : " << ptrVecTy << ", tensor<" << block
     << "xi32>\n";
  os << "  tt.store %co, %rval, %mask : " << ptrVecTy << "\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonReductionDot1D(const fir::fnacc::ElementwiseKernel &k,
                                     int64_t block, StringRef kernelName,
                                     llvm::raw_ostream &os) {
  assert(k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D &&
         "expected ReductionDot1D kernel");
  assert(k.readArrays.size() == 2 &&
         "dot reduction requires exactly two read arrays");

  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  // ABI:
  //
  //   %a        read array 0
  //   %b        read array 1
  //   %partials one output scalar per Triton program
  //   %n        number of elements
  //
  // Runtime recursively reduces %partials on the GPU and copies back only the
  // final scalar.
  //
  // NOTE: the textual tt.reduce form is Triton-version-sensitive. If your
  // Triton build uses a slightly different printed form, adjust this block
  // while preserving the function ABI.
  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %partials: " << ptrTy
     << ", %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";

  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";

  os << "  %base_s = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %base_s, %rng : tensor<" << block << "xi32>\n";

  os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %n_s : tensor<" << block << "xi32>\n";

  os << "  %ap = tt.splat %a : " << ptrTy << " -> tensor<" << block << "x"
     << ptrTy << ">\n";
  os << "  %ao = tt.addptr %ap, %offs : tensor<" << block << "x" << ptrTy
     << ">, tensor<" << block << "xi32>\n";
  os << "  %av = tt.load %ao, %mask : tensor<" << block << "x" << ptrTy
     << ">\n";

  os << "  %bp = tt.splat %b : " << ptrTy << " -> tensor<" << block << "x"
     << ptrTy << ">\n";
  os << "  %bo = tt.addptr %bp, %offs : tensor<" << block << "x" << ptrTy
     << ">, tensor<" << block << "xi32>\n";
  os << "  %bv = tt.load %bo, %mask : tensor<" << block << "x" << ptrTy
     << ">\n";

  os << "  %prod = arith.mulf %av, %bv : tensor<" << block << "x" << elemTy
     << ">\n";

  os << "  %zero = arith.constant 0.000000e+00 : " << elemTy << "\n";
  os << "  %zero_s = tt.splat %zero : " << elemTy << " -> tensor<" << block
     << "x" << elemTy << ">\n";

  os << "  %safe = arith.select %mask, %prod, %zero_s : tensor<" << block
     << "xi1>, tensor<" << block << "x" << elemTy << ">\n";

  os << "  %sum = \"tt.reduce\"(%safe) ({\n";
  os << "  ^bb0(%lhs: " << elemTy << ", %rhs: " << elemTy << "):\n";
  os << "    %r = arith.addf %lhs, %rhs : " << elemTy << "\n";
  os << "    \"tt.reduce.return\"(%r) : (" << elemTy << ") -> ()\n";
  os << "  }) {axis = 0 : i32} : (tensor<" << block << "x" << elemTy << ">) -> "
     << elemTy << "\n";

  os << "  %outp = tt.addptr %partials, %pid : " << ptrTy << ", i32\n";
  os << "  tt.store %outp, %sum : " << ptrTy << "\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonExpr2D(const fir::fnacc::ElementwiseKernel &k,
                             int64_t blockX, int64_t blockY,
                             StringRef kernelName, llvm::raw_ostream &os) {
  assert(k.expression && "Expr2D kernel has no expression tree");
  assert(!k.readArrays.empty() && k.readArrays.size() <= 3 &&
         "Expr2D requires one to three read arrays");

  int64_t block = blockX * blockY;
  std::string ptrTy = ptrType(k.elementType);
  std::string ptrVecTy = ptrTensorType(block, k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(";

  for (unsigned i = 0; i < k.readArrays.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << "%read" << i << ": " << ptrTy;
  }

  os << ", %c: " << ptrTy;

  for (unsigned i = 0; i < k.scalarRefs.size(); ++i)
    os << ", %scalar" << i << ": " << elemTy;

  os << ", %n: i32, %m: i32) attributes {noinline = false} {\n";

  os << "  %pid_x = tt.get_program_id x : i32\n";
  os << "  %pid_y = tt.get_program_id y : i32\n";
  os << "  %bx = arith.constant " << blockX << " : i32\n";
  os << "  %by = arith.constant " << blockY << " : i32\n";
  os << "  %base_x = arith.muli %pid_x, %bx : i32\n";
  os << "  %base_y = arith.muli %pid_y, %by : i32\n";
  os << "  %r = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";
  os << "  %bx_s = tt.splat %bx : i32 -> tensor<" << block << "xi32>\n";
  os << "  %local_i = arith.remui %r, %bx_s : tensor<" << block << "xi32>\n";
  os << "  %local_j = arith.divui %r, %bx_s : tensor<" << block << "xi32>\n";
  os << "  %base_x_s = tt.splat %base_x : i32 -> tensor<" << block << "xi32>\n";
  os << "  %base_y_s = tt.splat %base_y : i32 -> tensor<" << block << "xi32>\n";
  os << "  %ix = arith.addi %base_x_s, %local_i : tensor<" << block
     << "xi32>\n";
  os << "  %jy = arith.addi %base_y_s, %local_j : tensor<" << block
     << "xi32>\n";
  os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %m_s = tt.splat %m : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask_x = arith.cmpi slt, %ix, %n_s : tensor<" << block << "xi32>\n";
  os << "  %mask_y = arith.cmpi slt, %jy, %m_s : tensor<" << block << "xi32>\n";
  os << "  %mask = arith.andi %mask_x, %mask_y : tensor<" << block << "xi1>\n";
  os << "  %jy_n = arith.muli %jy, %n_s : tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %ix, %jy_n : tensor<" << block << "xi32>\n";

  if (k.readArrays.size() >= 1)
    emitLoad1D("%read0", "%read0v", block, k.elementType, os);
  if (k.readArrays.size() >= 2)
    emitLoad1D("%read1", "%read1v", block, k.elementType, os);
  if (k.readArrays.size() >= 3)
    emitLoad1D("%read2", "%read2v", block, k.elementType, os);

  ExprTritonEmitterState state;
  state.block = block;
  state.scalarSplatEmitted.resize(k.scalarRefs.size(), false);
  state.scalarSplatNames.resize(k.scalarRefs.size());

  std::string result = emitExprVector(k, *k.expression, state, os);

  os << "  %cp = tt.splat %c : " << ptrTy << " -> " << ptrVecTy << "\n";
  os << "  %co = tt.addptr %cp, %offs : " << ptrVecTy << ", tensor<" << block
     << "xi32>\n";
  os << "  tt.store %co, " << result << ", %mask : " << ptrVecTy << "\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonMatMul2DDot(const fir::fnacc::ElementwiseKernel &k,
                                  int64_t blockM, int64_t blockN,
                                  int64_t blockK, StringRef kernelName,
                                  llvm::raw_ostream &os) {

  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy
     << ", "
        "%n: i32, %m: i32, %k: i32) attributes {noinline = false} {\n";

  os << "  %pid_m = tt.get_program_id x : i32\n";
  os << "  %pid_n = tt.get_program_id y : i32\n";
  os << "  %bm = arith.constant " << blockM << " : i32\n";
  os << "  %bn = arith.constant " << blockN << " : i32\n";
  os << "  %bk = arith.constant " << blockK << " : i32\n";
  os << "  %base_m = arith.muli %pid_m, %bm : i32\n";
  os << "  %base_n = arith.muli %pid_n, %bn : i32\n";

  os << "  %offs_m0 = tt.make_range {start = 0 : i32, end = " << blockM
     << " : i32} : tensor<" << blockM << "xi32>\n";
  os << "  %offs_n0 = tt.make_range {start = 0 : i32, end = " << blockN
     << " : i32} : tensor<" << blockN << "xi32>\n";
  os << "  %offs_k0 = tt.make_range {start = 0 : i32, end = " << blockK
     << " : i32} : tensor<" << blockK << "xi32>\n";

  os << "  %base_m_s = tt.splat %base_m : i32 -> tensor<" << blockM
     << "xi32>\n";
  os << "  %base_n_s = tt.splat %base_n : i32 -> tensor<" << blockN
     << "xi32>\n";
  os << "  %offs_m = arith.addi %base_m_s, %offs_m0 : tensor<" << blockM
     << "xi32>\n";
  os << "  %offs_n = arith.addi %base_n_s, %offs_n0 : tensor<" << blockN
     << "xi32>\n";

  os << "  %n_s_m = tt.splat %n : i32 -> tensor<" << blockM << "xi32>\n";
  os << "  %m_s_n = tt.splat %m : i32 -> tensor<" << blockN << "xi32>\n";
  os << "  %mask_m = arith.cmpi slt, %offs_m, %n_s_m : tensor<" << blockM
     << "xi32>\n";
  os << "  %mask_n = arith.cmpi slt, %offs_n, %m_s_n : tensor<" << blockN
     << "xi32>\n";

  os << "  %zero = arith.constant 0.000000e+00 : " << elemTy << "\n";
  os << "  %acc0 = tt.splat %zero : " << elemTy << " -> tensor<" << blockM
     << "x" << blockN << "x" << elemTy << ">\n";

  os << "  %c0_idx = arith.constant 0 : index\n";
  os << "  %bk_idx = arith.constant " << blockK << " : index\n";
  os << "  %k_idx = arith.index_cast %k : i32 to index\n";

  os << "  %acc = scf.for %kk_idx = %c0_idx to %k_idx step %bk_idx "
        "iter_args(%acc_body = %acc0) -> (tensor<"
     << blockM << "x" << blockN << "x" << elemTy << ">) {\n";

  os << "    %kk_body = arith.index_cast %kk_idx : index to i32\n";
  os << "    %kk_s = tt.splat %kk_body : i32 -> tensor<" << blockK << "xi32>\n";
  os << "    %offs_k = arith.addi %kk_s, %offs_k0 : tensor<" << blockK
     << "xi32>\n";

  os << "    %offs_m_e = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<"
     << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n";
  os << "    %offs_k_e_a = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<"
     << blockK << "xi32> -> tensor<1x" << blockK << "xi32>\n";
  os << "    %offs_m_b = tt.broadcast %offs_m_e : tensor<" << blockM
     << "x1xi32> -> tensor<" << blockM << "x" << blockK << "xi32>\n";
  os << "    %offs_k_b_a = tt.broadcast %offs_k_e_a : tensor<1x" << blockK
     << "xi32> -> tensor<" << blockM << "x" << blockK << "xi32>\n";
  os << "    %n_s_a = tt.splat %n : i32 -> tensor<" << blockM << "x" << blockK
     << "xi32>\n";
  os << "    %a_k_n = arith.muli %offs_k_b_a, %n_s_a : tensor<" << blockM << "x"
     << blockK << "xi32>\n";
  os << "    %a_offsets = arith.addi %offs_m_b, %a_k_n : tensor<" << blockM
     << "x" << blockK << "xi32>\n";

  os << "    %offs_k_e_b = tt.expand_dims %offs_k {axis = 1 : i32} : tensor<"
     << blockK << "xi32> -> tensor<" << blockK << "x1xi32>\n";
  os << "    %offs_n_e = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<"
     << blockN << "xi32> -> tensor<1x" << blockN << "xi32>\n";
  os << "    %offs_k_b_b = tt.broadcast %offs_k_e_b : tensor<" << blockK
     << "x1xi32> -> tensor<" << blockK << "x" << blockN << "xi32>\n";
  os << "    %offs_n_b = tt.broadcast %offs_n_e : tensor<1x" << blockN
     << "xi32> -> tensor<" << blockK << "x" << blockN << "xi32>\n";
  os << "    %k_s_b = tt.splat %k : i32 -> tensor<" << blockK << "x" << blockN
     << "xi32>\n";
  os << "    %b_n_k = arith.muli %offs_n_b, %k_s_b : tensor<" << blockK << "x"
     << blockN << "xi32>\n";
  os << "    %b_offsets = arith.addi %offs_k_b_b, %b_n_k : tensor<" << blockK
     << "x" << blockN << "xi32>\n";

  os << "    %k_s_k = tt.splat %k : i32 -> tensor<" << blockK << "xi32>\n";
  os << "    %mask_k = arith.cmpi slt, %offs_k, %k_s_k : tensor<" << blockK
     << "xi32>\n";

  os << "    %mask_m_e = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<"
     << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n";
  os << "    %mask_k_e_a = tt.expand_dims %mask_k {axis = 0 : i32} : tensor<"
     << blockK << "xi1> -> tensor<1x" << blockK << "xi1>\n";
  os << "    %mask_m_b = tt.broadcast %mask_m_e : tensor<" << blockM
     << "x1xi1> -> tensor<" << blockM << "x" << blockK << "xi1>\n";
  os << "    %mask_k_b_a = tt.broadcast %mask_k_e_a : tensor<1x" << blockK
     << "xi1> -> tensor<" << blockM << "x" << blockK << "xi1>\n";
  os << "    %mask_a = arith.andi %mask_m_b, %mask_k_b_a : tensor<" << blockM
     << "x" << blockK << "xi1>\n";

  os << "    %mask_k_e_b = tt.expand_dims %mask_k {axis = 1 : i32} : tensor<"
     << blockK << "xi1> -> tensor<" << blockK << "x1xi1>\n";
  os << "    %mask_n_e = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<"
     << blockN << "xi1> -> tensor<1x" << blockN << "xi1>\n";
  os << "    %mask_k_b_b = tt.broadcast %mask_k_e_b : tensor<" << blockK
     << "x1xi1> -> tensor<" << blockK << "x" << blockN << "xi1>\n";
  os << "    %mask_n_b = tt.broadcast %mask_n_e : tensor<1x" << blockN
     << "xi1> -> tensor<" << blockK << "x" << blockN << "xi1>\n";
  os << "    %mask_b = arith.andi %mask_k_b_b, %mask_n_b : tensor<" << blockK
     << "x" << blockN << "xi1>\n";

  os << "    %a_base = tt.splat %a : " << ptrTy << " -> tensor<" << blockM
     << "x" << blockK << "x" << ptrTy << ">\n";
  os << "    %b_base = tt.splat %b : " << ptrTy << " -> tensor<" << blockK
     << "x" << blockN << "x" << ptrTy << ">\n";
  os << "    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<" << blockM << "x"
     << blockK << "x" << ptrTy << ">, tensor<" << blockM << "x" << blockK
     << "xi32>\n";
  os << "    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<" << blockK << "x"
     << blockN << "x" << ptrTy << ">, tensor<" << blockK << "x" << blockN
     << "xi32>\n";
  os << "    %a_tile = tt.load %a_ptrs, %mask_a : tensor<" << blockM << "x"
     << blockK << "x" << ptrTy << ">\n";
  os << "    %b_tile = tt.load %b_ptrs, %mask_b : tensor<" << blockK << "x"
     << blockN << "x" << ptrTy << ">\n";

  os << "    %acc_next = tt.dot %a_tile, %b_tile, %acc_body "
        "{inputPrecision = 0 : i32} : tensor<"
     << blockM << "x" << blockK << "x" << elemTy << "> * tensor<" << blockK
     << "x" << blockN << "x" << elemTy << "> -> tensor<" << blockM << "x"
     << blockN << "x" << elemTy << ">\n";
  os << "    scf.yield %acc_next : tensor<" << blockM << "x" << blockN << "x"
     << elemTy << ">\n";
  os << "  }\n";

  os << "  %offs_m_e_c = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<"
     << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n";
  os << "  %offs_n_e_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<"
     << blockN << "xi32> -> tensor<1x" << blockN << "xi32>\n";
  os << "  %offs_m_b_c = tt.broadcast %offs_m_e_c : tensor<" << blockM
     << "x1xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";
  os << "  %offs_n_b_c = tt.broadcast %offs_n_e_c : tensor<1x" << blockN
     << "xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";
  os << "  %n_s_c = tt.splat %n : i32 -> tensor<" << blockM << "x" << blockN
     << "xi32>\n";
  os << "  %c_j_n = arith.muli %offs_n_b_c, %n_s_c : tensor<" << blockM << "x"
     << blockN << "xi32>\n";
  os << "  %c_offsets = arith.addi %offs_m_b_c, %c_j_n : tensor<" << blockM
     << "x" << blockN << "xi32>\n";

  os << "  %mask_m_e_c = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<"
     << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n";
  os << "  %mask_n_e_c = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<"
     << blockN << "xi1> -> tensor<1x" << blockN << "xi1>\n";
  os << "  %mask_m_b_c = tt.broadcast %mask_m_e_c : tensor<" << blockM
     << "x1xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";
  os << "  %mask_n_b_c = tt.broadcast %mask_n_e_c : tensor<1x" << blockN
     << "xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";
  os << "  %mask_c = arith.andi %mask_m_b_c, %mask_n_b_c : tensor<" << blockM
     << "x" << blockN << "xi1>\n";

  os << "  %c_base = tt.splat %c : " << ptrTy << " -> tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">\n";
  os << "  %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">, tensor<" << blockM << "x" << blockN
     << "xi32>\n";
  os << "  tt.store %c_ptrs, %acc, %mask_c : tensor<" << blockM << "x" << blockN
     << "x" << ptrTy << ">\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonMatMul2DF32(const fir::fnacc::ElementwiseKernel &k,
                                  int64_t blockM, int64_t blockN,
                                  int64_t blockK, StringRef kernelName,
                                  llvm::raw_ostream &os) {
  assert(k.elementType == fir::fnacc::ElementType::F32);
  emitTritonMatMul2DDot(k, blockM, blockN, blockK, kernelName, os);
}

static void emitTritonMatMul2DF64Dot(const fir::fnacc::ElementwiseKernel &k,
                                     int64_t blockM, int64_t blockN,
                                     int64_t blockK, StringRef kernelName,
                                     llvm::raw_ostream &os) {
  assert(k.elementType == fir::fnacc::ElementType::F64);
  emitTritonMatMul2DDot(k, blockM, blockN, blockK, kernelName, os);
}

static void emitTritonMatMul2DF64Reduce(const fir::fnacc::ElementwiseKernel &k,
                                        int64_t blockM, int64_t blockN,
                                        int64_t blockK, StringRef kernelName,
                                        llvm::raw_ostream &os) {
  assert(k.elementType == fir::fnacc::ElementType::F64 &&
         "f64 matmul emitter requires f64 kernel");

  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy
     << ", %n: i32, %m: i32, %k: i32) attributes {noinline = false} {\n";

  os << "  %pid_m = tt.get_program_id x : i32\n";
  os << "  %pid_n = tt.get_program_id y : i32\n";

  os << "  %bm = arith.constant " << blockM << " : i32\n";
  os << "  %bn = arith.constant " << blockN << " : i32\n";
  os << "  %bk = arith.constant " << blockK << " : i32\n";

  os << "  %base_m = arith.muli %pid_m, %bm : i32\n";
  os << "  %base_n = arith.muli %pid_n, %bn : i32\n";

  os << "  %offs_m0 = tt.make_range {start = 0 : i32, end = " << blockM
     << " : i32} : tensor<" << blockM << "xi32>\n";
  os << "  %offs_n0 = tt.make_range {start = 0 : i32, end = " << blockN
     << " : i32} : tensor<" << blockN << "xi32>\n";
  os << "  %offs_k0 = tt.make_range {start = 0 : i32, end = " << blockK
     << " : i32} : tensor<" << blockK << "xi32>\n";

  os << "  %base_m_s = tt.splat %base_m : i32 -> tensor<" << blockM
     << "xi32>\n";
  os << "  %base_n_s = tt.splat %base_n : i32 -> tensor<" << blockN
     << "xi32>\n";

  os << "  %offs_m = arith.addi %base_m_s, %offs_m0 : tensor<" << blockM
     << "xi32>\n";
  os << "  %offs_n = arith.addi %base_n_s, %offs_n0 : tensor<" << blockN
     << "xi32>\n";

  os << "  %n_s_m = tt.splat %n : i32 -> tensor<" << blockM << "xi32>\n";
  os << "  %m_s_n = tt.splat %m : i32 -> tensor<" << blockN << "xi32>\n";

  os << "  %mask_m = arith.cmpi slt, %offs_m, %n_s_m : tensor<" << blockM
     << "xi32>\n";
  os << "  %mask_n = arith.cmpi slt, %offs_n, %m_s_n : tensor<" << blockN
     << "xi32>\n";

  os << "  %zero = arith.constant 0.000000e+00 : f64\n";
  os << "  %acc0 = tt.splat %zero : f64 -> tensor<" << blockM << "x" << blockN
     << "xf64>\n";

  os << "  %c0_idx = arith.constant 0 : index\n";
  os << "  %bk_idx = arith.constant " << blockK << " : index\n";
  os << "  %k_idx = arith.index_cast %k : i32 to index\n";

  os << "  %acc = scf.for %kk_idx = %c0_idx to %k_idx step %bk_idx "
        "iter_args(%acc_body = %acc0) -> (tensor<"
     << blockM << "x" << blockN << "xf64>) {\n";

  os << "    %kk = arith.index_cast %kk_idx : index to i32\n";
  os << "    %kk_s = tt.splat %kk : i32 -> tensor<" << blockK << "xi32>\n";
  os << "    %offs_k = arith.addi %kk_s, %offs_k0 : tensor<" << blockK
     << "xi32>\n";

  os << "    %k_s_k = tt.splat %k : i32 -> tensor<" << blockK << "xi32>\n";
  os << "    %mask_k = arith.cmpi slt, %offs_k, %k_s_k : tensor<" << blockK
     << "xi32>\n";

  // A offsets: A(i, p), column-major offset = i + p * n.
  os << "    %offs_m_e_a = tt.expand_dims %offs_m {axis = 1 : i32} "
        ": tensor<"
     << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n";
  os << "    %offs_k_e_a = tt.expand_dims %offs_k {axis = 0 : i32} "
        ": tensor<"
     << blockK << "xi32> -> tensor<1x" << blockK << "xi32>\n";
  os << "    %offs_m_b_a = tt.broadcast %offs_m_e_a : tensor<" << blockM
     << "x1xi32> -> tensor<" << blockM << "x" << blockK << "xi32>\n";
  os << "    %offs_k_b_a = tt.broadcast %offs_k_e_a : tensor<1x" << blockK
     << "xi32> -> tensor<" << blockM << "x" << blockK << "xi32>\n";
  os << "    %n_s_a = tt.splat %n : i32 -> tensor<" << blockM << "x" << blockK
     << "xi32>\n";
  os << "    %a_col = arith.muli %offs_k_b_a, %n_s_a : tensor<" << blockM << "x"
     << blockK << "xi32>\n";
  os << "    %a_offsets = arith.addi %offs_m_b_a, %a_col : tensor<" << blockM
     << "x" << blockK << "xi32>\n";

  // B offsets emitted as B(j, p) logical tensor N x K:
  // B(p,j), column-major offset = p + j * k.
  os << "    %offs_n_e_b = tt.expand_dims %offs_n {axis = 1 : i32} "
        ": tensor<"
     << blockN << "xi32> -> tensor<" << blockN << "x1xi32>\n";
  os << "    %offs_k_e_b = tt.expand_dims %offs_k {axis = 0 : i32} "
        ": tensor<"
     << blockK << "xi32> -> tensor<1x" << blockK << "xi32>\n";
  os << "    %offs_n_b_b = tt.broadcast %offs_n_e_b : tensor<" << blockN
     << "x1xi32> -> tensor<" << blockN << "x" << blockK << "xi32>\n";
  os << "    %offs_k_b_b = tt.broadcast %offs_k_e_b : tensor<1x" << blockK
     << "xi32> -> tensor<" << blockN << "x" << blockK << "xi32>\n";
  os << "    %k_s_b = tt.splat %k : i32 -> tensor<" << blockN << "x" << blockK
     << "xi32>\n";
  os << "    %b_col = arith.muli %offs_n_b_b, %k_s_b : tensor<" << blockN << "x"
     << blockK << "xi32>\n";
  os << "    %b_offsets = arith.addi %offs_k_b_b, %b_col : tensor<" << blockN
     << "x" << blockK << "xi32>\n";

  // Masks.
  os << "    %mask_m_e_a = tt.expand_dims %mask_m {axis = 1 : i32} "
        ": tensor<"
     << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n";
  os << "    %mask_k_e_a = tt.expand_dims %mask_k {axis = 0 : i32} "
        ": tensor<"
     << blockK << "xi1> -> tensor<1x" << blockK << "xi1>\n";
  os << "    %mask_m_b_a = tt.broadcast %mask_m_e_a : tensor<" << blockM
     << "x1xi1> -> tensor<" << blockM << "x" << blockK << "xi1>\n";
  os << "    %mask_k_b_a = tt.broadcast %mask_k_e_a : tensor<1x" << blockK
     << "xi1> -> tensor<" << blockM << "x" << blockK << "xi1>\n";
  os << "    %mask_a = arith.andi %mask_m_b_a, %mask_k_b_a : tensor<" << blockM
     << "x" << blockK << "xi1>\n";

  os << "    %mask_n_e_b = tt.expand_dims %mask_n {axis = 1 : i32} "
        ": tensor<"
     << blockN << "xi1> -> tensor<" << blockN << "x1xi1>\n";
  os << "    %mask_k_e_b = tt.expand_dims %mask_k {axis = 0 : i32} "
        ": tensor<"
     << blockK << "xi1> -> tensor<1x" << blockK << "xi1>\n";
  os << "    %mask_n_b_b = tt.broadcast %mask_n_e_b : tensor<" << blockN
     << "x1xi1> -> tensor<" << blockN << "x" << blockK << "xi1>\n";
  os << "    %mask_k_b_b = tt.broadcast %mask_k_e_b : tensor<1x" << blockK
     << "xi1> -> tensor<" << blockN << "x" << blockK << "xi1>\n";
  os << "    %mask_b = arith.andi %mask_n_b_b, %mask_k_b_b : tensor<" << blockN
     << "x" << blockK << "xi1>\n";

  // Loads.
  os << "    %a_base = tt.splat %a : " << ptrTy << " -> tensor<" << blockM
     << "x" << blockK << "x" << ptrTy << ">\n";
  os << "    %b_base = tt.splat %b : " << ptrTy << " -> tensor<" << blockN
     << "x" << blockK << "x" << ptrTy << ">\n";

  os << "    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<" << blockM << "x"
     << blockK << "x" << ptrTy << ">, tensor<" << blockM << "x" << blockK
     << "xi32>\n";
  os << "    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<" << blockN << "x"
     << blockK << "x" << ptrTy << ">, tensor<" << blockN << "x" << blockK
     << "xi32>\n";

  os << "    %a_tile = tt.load %a_ptrs, %mask_a : tensor<" << blockM << "x"
     << blockK << "x" << ptrTy << ">\n";
  os << "    %b_tile = tt.load %b_ptrs, %mask_b : tensor<" << blockN << "x"
     << blockK << "x" << ptrTy << ">\n";

  // A: M x K -> M x 1 x K -> M x N x K.
  os << "    %a_e = tt.expand_dims %a_tile {axis = 1 : i32} : tensor<" << blockM
     << "x" << blockK << "xf64> -> tensor<" << blockM << "x1x" << blockK
     << "xf64>\n";
  os << "    %a_b = tt.broadcast %a_e : tensor<" << blockM << "x1x" << blockK
     << "xf64> -> tensor<" << blockM << "x" << blockN << "x" << blockK
     << "xf64>\n";

  // B: N x K -> 1 x N x K -> M x N x K.
  os << "    %b_e = tt.expand_dims %b_tile {axis = 0 : i32} : tensor<" << blockN
     << "x" << blockK << "xf64> -> tensor<1x" << blockN << "x" << blockK
     << "xf64>\n";
  os << "    %b_b = tt.broadcast %b_e : tensor<1x" << blockN << "x" << blockK
     << "xf64> -> tensor<" << blockM << "x" << blockN << "x" << blockK
     << "xf64>\n";

  os << "    %prod = arith.mulf %a_b, %b_b : tensor<" << blockM << "x" << blockN
     << "x" << blockK << "xf64>\n";

  // Reduce over K dimension.
  os << "    %partial = \"tt.reduce\"(%prod) ({\n";
  os << "    ^bb0(%lhs: f64, %rhs: f64):\n";
  os << "      %r = arith.addf %lhs, %rhs : f64\n";
  os << "      \"tt.reduce.return\"(%r) : (f64) -> ()\n";
  os << "    }) {axis = 2 : i32} : (tensor<" << blockM << "x" << blockN << "x"
     << blockK << "xf64>) -> tensor<" << blockM << "x" << blockN << "xf64>\n";

  os << "    %acc_next = arith.addf %acc_body, %partial : tensor<" << blockM
     << "x" << blockN << "xf64>\n";

  os << "    scf.yield %acc_next : tensor<" << blockM << "x" << blockN
     << "xf64>\n";

  os << "  }\n";

  // Store C(i,j), offset = i + j * n
  os << "  %offs_m_e_c = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<"
     << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n";

  os << "  %offs_n_e_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<"
     << blockN << "xi32> -> tensor<1x" << blockN << "xi32>\n";

  os << "  %offs_m_b_c = tt.broadcast %offs_m_e_c : tensor<" << blockM
     << "x1xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";

  os << "  %offs_n_b_c = tt.broadcast %offs_n_e_c : tensor<1x" << blockN
     << "xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";

  os << "  %n_s_c = tt.splat %n : i32 -> tensor<" << blockM << "x" << blockN
     << "xi32>\n";

  os << "  %c_j_n = arith.muli %offs_n_b_c, %n_s_c : tensor<" << blockM << "x"
     << blockN << "xi32>\n";

  os << "  %c_offsets = arith.addi %offs_m_b_c, %c_j_n : tensor<" << blockM
     << "x" << blockN << "xi32>\n";

  os << "  %mask_m_e_c = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<"
     << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n";

  os << "  %mask_n_e_c = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<"
     << blockN << "xi1> -> tensor<1x" << blockN << "xi1>\n";

  os << "  %mask_m_b_c = tt.broadcast %mask_m_e_c : tensor<" << blockM
     << "x1xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";

  os << "  %mask_n_b_c = tt.broadcast %mask_n_e_c : tensor<1x" << blockN
     << "xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";

  os << "  %mask_c = arith.andi %mask_m_b_c, %mask_n_b_c : tensor<" << blockM
     << "x" << blockN << "xi1>\n";

  os << "  %c_base = tt.splat %c : " << ptrTy << " -> tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">\n";

  os << "  %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">, tensor<" << blockM << "x" << blockN
     << "xi32>\n";

  os << "  tt.store %c_ptrs, %acc, %mask_c : tensor<" << blockM << "x" << blockN
     << "x" << ptrTy << ">\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonMatMul2DF64FMA(const fir::fnacc::ElementwiseKernel &k,
                                     int64_t blockM, int64_t blockN,
                                     int64_t blockK, StringRef kernelName,
                                     llvm::raw_ostream &os) {
  assert(k.elementType == fir::fnacc::ElementType::F64 &&
         "f64 matmul emitter requires f64 kernel");

  std::string ptrTy = ptrType(k.elementType);
  std::string elemTy = ttElementType(k.elementType).str();

  os << "tt.func @" << kernelName << "(%a: " << ptrTy << ", %b: " << ptrTy
     << ", %c: " << ptrTy
     << ", %n: i32, %m: i32, %k: i32) attributes {noinline = false} {\n";

  os << "  %pid_m = tt.get_program_id x : i32\n";
  os << "  %pid_n = tt.get_program_id y : i32\n";

  os << "  %bm = arith.constant " << blockM << " : i32\n";
  os << "  %bn = arith.constant " << blockN << " : i32\n";
  os << "  %bk = arith.constant " << blockK << " : i32\n";

  os << "  %base_m = arith.muli %pid_m, %bm : i32\n";
  os << "  %base_n = arith.muli %pid_n, %bn : i32\n";

  os << "  %offs_m0 = tt.make_range {start = 0 : i32, end = " << blockM
     << " : i32} : tensor<" << blockM << "xi32>\n";
  os << "  %offs_n0 = tt.make_range {start = 0 : i32, end = " << blockN
     << " : i32} : tensor<" << blockN << "xi32>\n";

  os << "  %base_m_s = tt.splat %base_m : i32 -> tensor<" << blockM
     << "xi32>\n";
  os << "  %base_n_s = tt.splat %base_n : i32 -> tensor<" << blockN
     << "xi32>\n";

  os << "  %offs_m = arith.addi %base_m_s, %offs_m0 : tensor<" << blockM
     << "xi32>\n";
  os << "  %offs_n = arith.addi %base_n_s, %offs_n0 : tensor<" << blockN
     << "xi32>\n";

  os << "  %n_s_m = tt.splat %n : i32 -> tensor<" << blockM << "xi32>\n";
  os << "  %m_s_n = tt.splat %m : i32 -> tensor<" << blockN << "xi32>\n";

  os << "  %mask_m = arith.cmpi slt, %offs_m, %n_s_m : tensor<" << blockM
     << "xi32>\n";
  os << "  %mask_n = arith.cmpi slt, %offs_n, %m_s_n : tensor<" << blockN
     << "xi32>\n";

  os << "  %zero = arith.constant 0.000000e+00 : f64\n";
  os << "  %acc0 = tt.splat %zero : f64 -> tensor<" << blockM << "x" << blockN
     << "xf64>\n";

  os << "  %c0_idx = arith.constant 0 : index\n";
  os << "  %bk_idx = arith.constant " << blockK << " : index\n";
  os << "  %k_idx = arith.index_cast %k : i32 to index\n";

  os << "  %acc = scf.for %kk_idx = %c0_idx to %k_idx step %bk_idx "
        "iter_args(%acc_body = %acc0) -> (tensor<"
     << blockM << "x" << blockN << "xf64>) {\n";

  os << "    %kk_base = arith.index_cast %kk_idx : index to i32\n";

  // Start each K-block accumulation from the incoming accumulator.
  std::string accPrev = "%acc_body";

  for (int64_t q = 0; q < blockK; ++q) {
    std::string suffix = std::to_string(q);

    os << "    %q" << suffix << " = arith.constant " << q << " : i32\n";
    os << "    %kk" << suffix << " = arith.addi %kk_base, %q" << suffix
       << " : i32\n";

    // Scalar K mask for this unrolled K lane:
    //
    //   kk + q < k
    os << "    %mask_k_scalar" << suffix << " = arith.cmpi slt, %kk" << suffix
       << ", %k : i32\n";

    os << "    %mask_k_m" << suffix << " = tt.splat %mask_k_scalar" << suffix
       << " : i1 -> tensor<" << blockM << "xi1>\n";
    os << "    %mask_k_n" << suffix << " = tt.splat %mask_k_scalar" << suffix
       << " : i1 -> tensor<" << blockN << "xi1>\n";

    os << "    %mask_a" << suffix << " = arith.andi %mask_m, %mask_k_m"
       << suffix << " : tensor<" << blockM << "xi1>\n";
    os << "    %mask_b" << suffix << " = arith.andi %mask_n, %mask_k_n"
       << suffix << " : tensor<" << blockN << "xi1>\n";

    // A(i, kk+q), column-major offset:
    //
    //   i + (kk+q) * n
    os << "    %kk_m" << suffix << " = tt.splat %kk" << suffix
       << " : i32 -> tensor<" << blockM << "xi32>\n";
    os << "    %n_s_m_body" << suffix << " = tt.splat %n : i32 -> tensor<"
       << blockM << "xi32>\n";
    os << "    %a_col" << suffix << " = arith.muli %kk_m" << suffix
       << ", %n_s_m_body" << suffix << " : tensor<" << blockM << "xi32>\n";
    os << "    %a_offsets" << suffix << " = arith.addi %offs_m, %a_col"
       << suffix << " : tensor<" << blockM << "xi32>\n";

    // B(kk+q, j), column-major offset:
    //
    //   (kk+q) + j * k
    os << "    %kk_n" << suffix << " = tt.splat %kk" << suffix
       << " : i32 -> tensor<" << blockN << "xi32>\n";
    os << "    %k_s_n_body" << suffix << " = tt.splat %k : i32 -> tensor<"
       << blockN << "xi32>\n";
    os << "    %b_col" << suffix << " = arith.muli %offs_n, %k_s_n_body"
       << suffix << " : tensor<" << blockN << "xi32>\n";
    os << "    %b_offsets" << suffix << " = arith.addi %kk_n" << suffix
       << ", %b_col" << suffix << " : tensor<" << blockN << "xi32>\n";

    os << "    %a_base" << suffix << " = tt.splat %a : " << ptrTy
       << " -> tensor<" << blockM << "x" << ptrTy << ">\n";
    os << "    %b_base" << suffix << " = tt.splat %b : " << ptrTy
       << " -> tensor<" << blockN << "x" << ptrTy << ">\n";

    os << "    %a_ptrs" << suffix << " = tt.addptr %a_base" << suffix
       << ", %a_offsets" << suffix << " : tensor<" << blockM << "x" << ptrTy
       << ">, tensor<" << blockM << "xi32>\n";
    os << "    %b_ptrs" << suffix << " = tt.addptr %b_base" << suffix
       << ", %b_offsets" << suffix << " : tensor<" << blockN << "x" << ptrTy
       << ">, tensor<" << blockN << "xi32>\n";

    os << "    %a_vec" << suffix << " = tt.load %a_ptrs" << suffix
       << ", %mask_a" << suffix << " : tensor<" << blockM << "x" << ptrTy
       << ">\n";
    os << "    %b_vec" << suffix << " = tt.load %b_ptrs" << suffix
       << ", %mask_b" << suffix << " : tensor<" << blockN << "x" << ptrTy
       << ">\n";

    // Broadcast A(:, kk+q) and B(kk+q, :) to an M x N tile.
    os << "    %a_e" << suffix << " = tt.expand_dims %a_vec" << suffix
       << " {axis = 1 : i32} : tensor<" << blockM << "xf64> -> tensor<"
       << blockM << "x1xf64>\n";
    os << "    %b_e" << suffix << " = tt.expand_dims %b_vec" << suffix
       << " {axis = 0 : i32} : tensor<" << blockN << "xf64> -> tensor<1x"
       << blockN << "xf64>\n";

    os << "    %a_b" << suffix << " = tt.broadcast %a_e" << suffix
       << " : tensor<" << blockM << "x1xf64> -> tensor<" << blockM << "x"
       << blockN << "xf64>\n";
    os << "    %b_b" << suffix << " = tt.broadcast %b_e" << suffix
       << " : tensor<1x" << blockN << "xf64> -> tensor<" << blockM << "x"
       << blockN << "xf64>\n";

    std::string accNext = "%acc_fma" + suffix;

    // Explicit f64 FMA:
    //
    //   acc = A(:, kk+q) * B(kk+q, :) + acc
    //
    // This avoids materialising tensor<MxNxKxf64> and avoids tt.reduce.
    os << "    " << accNext << " = math.fma %a_b" << suffix << ", %b_b"
       << suffix << ", " << accPrev << " : tensor<" << blockM << "x" << blockN
       << "xf64>\n";

    accPrev = accNext;
  }

  os << "    scf.yield " << accPrev << " : tensor<" << blockM << "x" << blockN
     << "xf64>\n";

  os << "  }\n";

  // Store C(i,j), column-major offset:
  //
  //   i + j * n
  os << "  %offs_m_e_c = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<"
     << blockM << "xi32> -> tensor<" << blockM << "x1xi32>\n";

  os << "  %offs_n_e_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<"
     << blockN << "xi32> -> tensor<1x" << blockN << "xi32>\n";

  os << "  %offs_m_b_c = tt.broadcast %offs_m_e_c : tensor<" << blockM
     << "x1xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";

  os << "  %offs_n_b_c = tt.broadcast %offs_n_e_c : tensor<1x" << blockN
     << "xi32> -> tensor<" << blockM << "x" << blockN << "xi32>\n";

  os << "  %n_s_c = tt.splat %n : i32 -> tensor<" << blockM << "x" << blockN
     << "xi32>\n";

  os << "  %c_j_n = arith.muli %offs_n_b_c, %n_s_c : tensor<" << blockM << "x"
     << blockN << "xi32>\n";

  os << "  %c_offsets = arith.addi %offs_m_b_c, %c_j_n : tensor<" << blockM
     << "x" << blockN << "xi32>\n";

  os << "  %mask_m_e_c = tt.expand_dims %mask_m {axis = 1 : i32} : tensor<"
     << blockM << "xi1> -> tensor<" << blockM << "x1xi1>\n";

  os << "  %mask_n_e_c = tt.expand_dims %mask_n {axis = 0 : i32} : tensor<"
     << blockN << "xi1> -> tensor<1x" << blockN << "xi1>\n";

  os << "  %mask_m_b_c = tt.broadcast %mask_m_e_c : tensor<" << blockM
     << "x1xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";

  os << "  %mask_n_b_c = tt.broadcast %mask_n_e_c : tensor<1x" << blockN
     << "xi1> -> tensor<" << blockM << "x" << blockN << "xi1>\n";

  os << "  %mask_c = arith.andi %mask_m_b_c, %mask_n_b_c : tensor<" << blockM
     << "x" << blockN << "xi1>\n";

  os << "  %c_base = tt.splat %c : " << ptrTy << " -> tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">\n";

  os << "  %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<" << blockM << "x"
     << blockN << "x" << ptrTy << ">, tensor<" << blockM << "x" << blockN
     << "xi32>\n";

  os << "  tt.store %c_ptrs, %acc, %mask_c : tensor<" << blockM << "x" << blockN
     << "x" << ptrTy << ">\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static llvm::SmallVector<unsigned>
kernelParamSlotsForValue(const fir::fnacc::ElementwiseKernel &k, Value v) {
  llvm::SmallVector<unsigned> slots;

  for (unsigned i = 0; i < k.readArrays.size(); ++i)
    if (k.readArrays[i] == v)
      slots.push_back(i);

  if (k.writeArray == v)
    slots.push_back(k.readArrays.size());

  unsigned scalarBaseSlot = k.readArrays.size() + 1;
  for (unsigned i = 0; i < k.scalarRefs.size(); ++i)
    if (k.scalarRefs[i] == v)
      slots.push_back(scalarBaseSlot + i);

  return slots;
}

static bool isReductionMetadataPackSlot(fir::fnacc::LaunchOp launchOp,
                                        unsigned packIndex) {
  auto slotsAttr =
      launchOp->getAttrOfType<DenseI32ArrayAttr>("fnacc.reduction_slots");

  if (!slotsAttr)
    return false;

  for (int32_t slot : slotsAttr.asArrayRef()) {
    if (slot >= 0 && static_cast<unsigned>(slot) == packIndex)
      return true;
  }

  return false;
}

static void emitJsonDescriptor(
    fir::fnacc::LaunchOp launchOp, const fir::fnacc::ElementwiseKernel &k,
    int64_t blockX, int64_t blockY, int64_t blockZ, int32_t kernelId,
    int32_t ptxIndex, llvm::StringRef kernelName, int32_t tritonNumWarps,
    int32_t tritonThreadsPerWarp, int32_t tritonNumStages,
    int32_t cudaThreadsPerCTA, int32_t reductionStageId, llvm::raw_ostream &os,
    bool &firstKernel) {

  if (!firstKernel)
    os << ",\n";
  firstKernel = false;

  StringRef kindName = "binary";
  if (k.kind == fir::fnacc::ElementwiseKernelKind::Saxpy1D)
    kindName = "saxpy1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::Expr1D)
    kindName = "expr1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::Expr2D)
    kindName = "expr2d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D)
    kindName = "matmul2d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionSum1D)
    kindName = "reduction_sum1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D)
    kindName = "reduction_dot1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionProduct1D)
    kindName = "reduction_product1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMin1D)
    kindName = "reduction_min1d";
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMax1D)
    kindName = "reduction_max1d";

  std::string ptrJsonTy = jsonPtrType(k.elementType);
  std::string elemJsonTy = jsonElementType(k.elementType);

  os << "    {\n";
  os << "      \"id\": " << kernelId << ",\n";
  os << "      \"name\": \"" << kernelName << "\",\n";
  os << "      \"ptx_index\": " << ptxIndex << ",\n";
  os << "      \"ptx_file\": \"" << kernelName << ".ptx\",\n";
  os << "      \"kind\": \"" << kindName << "\",\n";
  os << "      \"rank\": " << k.rank << ",\n";
  os << "      \"tile\": [" << blockX << ", " << blockY << ", " << blockZ
     << "],\n";
  os << "      \"num_warps\": " << tritonNumWarps << ",\n";
  os << "      \"threads_per_warp\": " << tritonThreadsPerWarp << ",\n";
  os << "      \"num_ctas\": 1,\n";
  os << "      \"num_stages\": " << tritonNumStages << ",\n";
  os << "      \"cuda_threads_per_cta\": " << cudaThreadsPerCTA << ",\n";
  os << "      \"triton_hidden_ptr_args\": " << kTritonHiddenPtrArgs << ",\n";

  if (reductionStageId >= 0)
    os << "      \"reduction_stage_id\": " << reductionStageId << ",\n";

  if (k.rank == 2) {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", "
       << "\"cdiv(extent_y, tile_y)\", \"1\"],\n";
  } else {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  }

  bool isReduction =
      k.kind == fir::fnacc::ElementwiseKernelKind::ReductionSum1D ||
      k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D ||
      k.kind == fir::fnacc::ElementwiseKernelKind::ReductionProduct1D ||
      k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMin1D ||
      k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMax1D;

  if (isReduction)
    os << "      \"reduction_op\": \""
       << reductionOperatorName(k.reductionOperator) << "\",\n";

  os << "      \"params\": [\n";

  unsigned nextSlot = 0;

  if (isReduction) {
    // Reduction kernel ABI:
    //
    //   reduction_sum1d:
    //     read0, partials, extent_x
    //
    //   reduction_dot1d:
    //     read0, read1, partials, extent_x
    //
    // The final scalar result is not a Triton kernel parameter. Runtime
    // lowering passes the scalar to the host runtime, which recursively
    // reduces the partials buffer with the reduction_stage_id kernel.
    for (unsigned i = 0; i < k.readArrays.size(); ++i) {
      if (i != 0)
        os << ",\n";

      os << "        {\"slot\": " << nextSlot++
         << ", \"role\": \"read\",     \"name\": \"read" << i
         << "\",    \"type\": \"" << ptrJsonTy << "\"}";
    }

    if (!k.readArrays.empty())
      os << ",\n";

    os << "        {\"slot\": " << nextSlot++
       << ", \"role\": \"partials\", \"name\": \"partials\", "
       << "\"type\": \"" << ptrJsonTy << "\"},\n";

    os << "        {\"slot\": " << nextSlot++
       << ", \"role\": \"extent_x\", \"name\": \"extent_x\", "
       << "\"type\": \"i32\"}\n";

    os << "      ],\n";

  } else {
    // Generic elementwise/matmul parameter metadata.
    for (unsigned i = 0; i < k.readArrays.size(); ++i) {
      if (i != 0)
        os << ",\n";

      os << "        {\"slot\": " << nextSlot++
         << ", \"role\": \"read\",     \"name\": \"read" << i
         << "\",    \"type\": \"" << ptrJsonTy << "\"}";
    }

    if (!k.readArrays.empty())
      os << ",\n";

    os << "        {\"slot\": " << nextSlot++
       << ", \"role\": \"write\",    \"name\": \"write\",    "
       << "\"type\": \"" << ptrJsonTy << "\"}";

    for (unsigned i = 0; i < k.scalarRefs.size(); ++i) {
      os << ",\n";
      os << "        {\"slot\": " << nextSlot++
         << ", \"role\": \"scalar\",   \"name\": \"scalar" << i
         << "\",  \"type\": \"" << elemJsonTy << "\"}";
    }

    os << ",\n";
    os << "        {\"slot\": " << nextSlot++
       << ", \"role\": \"extent_x\", \"name\": \"extent_x\", "
       << "\"type\": \"i32\"}";

    if (k.rank == 2) {
      os << ",\n";
      os << "        {\"slot\": " << nextSlot++
         << ", \"role\": \"extent_y\", \"name\": \"extent_y\", "
         << "\"type\": \"i32\"}";

      if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
        os << ",\n";
        os << "        {\"slot\": " << nextSlot++
           << ", \"role\": \"extent_k\", \"name\": \"extent_k\", "
           << "\"type\": \"i32\"}\n";
      } else {
        os << "\n";
      }
    } else {
      os << "\n";
    }

    os << "      ],\n";
  }

  os << "      \"pack\": [";

  auto packVars = launchOp.getPackVars();
  llvm::ArrayRef<int32_t> targets = launchOp.getPackTargets();

  bool firstPack = true;
  for (auto it : llvm::enumerate(packVars)) {
    unsigned packIndex = it.index();
    Value packValue = it.value();

    // Reduction scalars are carried through pack_vars only as launch metadata.
    // They are not Triton kernel arguments and should not appear in JSON pack
    // entries.
    if (isReductionMetadataPackSlot(launchOp, packIndex))
      continue;

    llvm::SmallVector<unsigned> slots = kernelParamSlotsForValue(k, packValue);
    if (slots.empty()) {
      launchOp.emitWarning()
          << "PACK variable #" << packIndex
          << " was not used by recognized Triton kernel body";
      continue;
    }

    if (packIndex >= targets.size()) {
      launchOp.emitWarning("PACK target list shorter than PACK var list");
      continue;
    }

    int32_t target = targets[packIndex];

    for (unsigned slot : slots) {
      if (!firstPack)
        os << ", ";
      firstPack = false;

      os << "{\"kernel_arg_slot\": " << slot << ", \"target\": " << target
         << ", \"target_name\": \"" << (target == 0 ? "host" : "device")
         << "\"}";
    }
  }

  os << "]\n";
  os << "    }";
}

static void emitJsonReductionStageDescriptor(
    fir::fnacc::ElementType type, fir::fnacc::ReductionOperator reductionOp,
    int64_t block, int32_t kernelId, int32_t ptxIndex,
    llvm::StringRef kernelName, int32_t tritonNumWarps,
    int32_t tritonThreadsPerWarp, int32_t tritonNumStages,
    int32_t cudaThreadsPerCTA, llvm::raw_ostream &os, bool &firstKernel) {
  if (!firstKernel)
    os << ",\n";
  firstKernel = false;

  std::string ptrJsonTy = jsonPtrType(type);

  os << "    {\n";
  os << "      \"id\": " << kernelId << ",\n";
  os << "      \"name\": \"" << kernelName << "\",\n";
  os << "      \"ptx_index\": " << ptxIndex << ",\n";
  os << "      \"ptx_file\": \"" << kernelName << ".ptx\",\n";
  os << "      \"kind\": \"reduction_stage1d\",\n";
  os << "      \"reduction_op\": \"" << reductionOperatorName(reductionOp)
     << "\",\n";
  os << "      \"rank\": 1,\n";
  os << "      \"tile\": [" << block << ", 1, 1],\n";
  os << "      \"num_warps\": " << tritonNumWarps << ",\n";
  os << "      \"threads_per_warp\": " << tritonThreadsPerWarp << ",\n";
  os << "      \"num_ctas\": 1,\n";
  os << "      \"num_stages\": " << tritonNumStages << ",\n";
  os << "      \"cuda_threads_per_cta\": " << cudaThreadsPerCTA << ",\n";
  os << "      \"triton_hidden_ptr_args\": " << kTritonHiddenPtrArgs << ",\n";
  os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  os << "      \"params\": [\n";
  os << "        {\"slot\": 0, \"role\": \"read\", "
        "\"name\": \"input\", \"type\": \""
     << ptrJsonTy << "\"},\n";
  os << "        {\"slot\": 1, \"role\": \"partials\", "
        "\"name\": \"output\", \"type\": \""
     << ptrJsonTy << "\"},\n";
  os << "        {\"slot\": 2, \"role\": \"extent_x\", "
        "\"name\": \"extent_x\", \"type\": \"i32\"}\n";
  os << "      ],\n";
  os << "      \"pack\": []\n";
  os << "    }";
}

struct FNACCLowerToTritonPass
    : public fir::fnacc::impl::FNACCLowerToTritonBase<FNACCLowerToTritonPass> {
  FNACCLowerToTritonPass() = default;

  FNACCLowerToTritonPass(llvm::StringRef ttirOutput, llvm::StringRef jsonOutput,
                         int32_t numWarps, int32_t threadsPerWarp,
                         int32_t numStages, llvm::StringRef f64MatmulStrategy) {
    this->ttirOutput = ttirOutput.str();
    this->jsonOutput = jsonOutput.str();
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numStages = numStages;
    this->f64MatmulStrategy = f64MatmulStrategy;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int32_t tritonNumWarps = this->numWarps;
    int32_t tritonThreadsPerWarp = this->threadsPerWarp;
    int32_t tritonNumStages = this->numStages;

    if (tritonNumWarps <= 0) {
      module.emitError("FNACC num-warps must be positive");
      signalPassFailure();
      return;
    }

    if (tritonThreadsPerWarp <= 0) {
      module.emitError("FNACC threads-per-warp must be positive");
      signalPassFailure();
      return;
    }

    if (tritonNumStages <= 0) {
      module.emitError("FNACC num-stages must be positive");
      signalPassFailure();
      return;
    }

    if (tritonThreadsPerWarp != 32) {
      module.emitError("FNACC currently supports exactly 32 threads per warp");
      signalPassFailure();
      return;
    }

    if (this->f64MatmulStrategy != "dot" &&
        this->f64MatmulStrategy != "reduce" &&
        this->f64MatmulStrategy != "fma") {
      module.emitError("FNACC f64-matmul-strategy must be dot, reduce, or fma");
      signalPassFailure();
      return;
    }

    bool hasSimpleElementwiseLaunch = false;
    bool hasReductionLaunch = false;
    bool hasMatmulLaunch = false;
    bool recognitionFailed = false;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);
      if (result.failed()) {
        launchOp.emitError("FNACC Triton cannot emit launch: ")
            << result.getFailure().reason;
        recognitionFailed = true;
        return;
      }

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();

      switch (k.kind) {
      case fir::fnacc::ElementwiseKernelKind::BinaryArrayArray:
      case fir::fnacc::ElementwiseKernelKind::Saxpy1D:
      case fir::fnacc::ElementwiseKernelKind::Expr1D:
      case fir::fnacc::ElementwiseKernelKind::Expr2D:
        hasSimpleElementwiseLaunch = true;
        break;

      case fir::fnacc::ElementwiseKernelKind::ReductionSum1D:
      case fir::fnacc::ElementwiseKernelKind::ReductionDot1D:
      case fir::fnacc::ElementwiseKernelKind::ReductionProduct1D:
      case fir::fnacc::ElementwiseKernelKind::ReductionMin1D:
      case fir::fnacc::ElementwiseKernelKind::ReductionMax1D:
        hasReductionLaunch = true;
        break;

      case fir::fnacc::ElementwiseKernelKind::MatMul2D:
        hasMatmulLaunch = true;
        break;
      }
    });

    if (recognitionFailed) {
      signalPassFailure();
      return;
    }

    if ((hasSimpleElementwiseLaunch || hasReductionLaunch) && hasMatmulLaunch &&
        tritonNumWarps != 1) {
      module.emitWarning()
          << "FNACC module contains both simple elementwise/reduction kernels "
             "and "
             "matmul kernels. Per-kernel TTIR splitting is required for true "
             "per-kernel num-warps. The FNACC wrapper should compile each "
             "emitted "
             "kernel separately.";
    }

    std::string ttirPath = this->ttirOutput;
    std::string jsonPath = this->jsonOutput;

    std::error_code ttirEc;
    llvm::raw_fd_ostream ttirOs(ttirPath, ttirEc);
    if (ttirEc) {
      module.emitError("cannot open FNACC TTIR output file: ") << ttirPath;
      signalPassFailure();
      return;
    }

    std::error_code jsonEc;
    llvm::raw_fd_ostream jsonOs(jsonPath, jsonEc);
    if (jsonEc) {
      module.emitError("cannot open FNACC JSON output file: ") << jsonPath;
      signalPassFailure();
      return;
    }

    jsonOs << "{\n";
    jsonOs << "  \"fnacc_schema_version\": 1,\n";
    jsonOs << "  \"kernels\": [\n";

    ttirOs << "module attributes {"
           << "\"ttg.num-warps\" = " << tritonNumWarps << " : i32, "
           << "\"ttg.num-ctas\" = 1 : i32, "
           << "\"ttg.num-stages\" = " << tritonNumStages << " : i32, "
           << "\"ttg.threads-per-warp\" = " << tritonThreadsPerWarp << " : i32"
           << "} {\n";

    bool firstKernel = true;
    int32_t fallbackId = 0;
    bool failed = false;

    int32_t emittedPtxIndex = 0;
    int32_t nextSyntheticKernelId = 0;
    int32_t scanFallbackId = 0;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      int32_t kernelId = getKernelId(launchOp, scanFallbackId++);
      nextSyntheticKernelId = std::max(nextSyntheticKernelId, kernelId + 1);
    });

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      if (failed)
        return;

      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);
      if (result.failed()) {
        launchOp.emitError("FNACC Triton recognition changed during emission: ")
            << result.getFailure().reason;
        failed = true;
        return;
      }

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();

      int32_t kernelId = getKernelId(launchOp, fallbackId);
      std::string kernelName = getKernelName(launchOp, kernelId);

      bool isReduction =
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionSum1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionProduct1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMin1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMax1D;
      int32_t reductionStageId = isReduction ? nextSyntheticKernelId++ : -1;
      std::string reductionStageName = kernelName + "_reduce_stage";

      llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

      int64_t blockX = 1024;
      int64_t blockY = 1;
      int64_t blockZ = 1;

      if (k.rank == 2) {
        blockX = tiles.size() >= 1 ? tiles[0] : 16;
        blockY = tiles.size() >= 2 ? tiles[1] : 16;

        if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
          blockZ =
              tiles.size() >= 3
                  ? tiles[2]
                  : (k.elementType == fir::fnacc::ElementType::F64 ? 8 : 32);

          if (k.elementType == fir::fnacc::ElementType::F64) {
            if (this->f64MatmulStrategy == "dot") {
              emitTritonMatMul2DF64Dot(k, blockX, blockY, blockZ, kernelName,
                                       ttirOs);
            } else if (this->f64MatmulStrategy == "fma") {
              emitTritonMatMul2DF64FMA(k, blockX, blockY, blockZ, kernelName,
                                       ttirOs);
            } else if (this->f64MatmulStrategy == "reduce") {
              emitTritonMatMul2DF64Reduce(k, blockX, blockY, blockZ, kernelName,
                                          ttirOs);
            } else {
              launchOp.emitError("unknown FNACC f64 matmul strategy: ")
                  << this->f64MatmulStrategy;
              failed = true;
              return;
            }
          } else {
            emitTritonMatMul2DF32(k, blockX, blockY, blockZ, kernelName,
                                  ttirOs);
          }

        } else if (k.kind == fir::fnacc::ElementwiseKernelKind::Expr2D) {
          blockZ = 1;
          emitTritonExpr2D(k, blockX, blockY, kernelName, ttirOs);
        } else {
          blockZ = 1;
          emitTriton2D(k, blockX, blockY, kernelName, ttirOs);
        }
      } else {
        blockX = tiles.empty() ? 1024 : tiles[0];
        blockY = 1;
        blockZ = 1;

        if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D) {
          emitTritonReductionDot1D(k, blockX, kernelName, ttirOs);
        } else if (isReduction) {
          emitTritonReduction1D(k, blockX, kernelName, ttirOs);
        } else if (k.kind == fir::fnacc::ElementwiseKernelKind::Expr1D) {
          emitTritonExpr1D(k, blockX, kernelName, ttirOs);
        } else if (k.kind == fir::fnacc::ElementwiseKernelKind::Saxpy1D) {
          emitTritonSaxpy1D(k, blockX, kernelName, ttirOs);
        } else {
          emitTriton1D(k, blockX, kernelName, ttirOs);
        }
      }

      if (isReduction) {
        emitTritonReductionStage1D(k.elementType, k.reductionOperator, blockX,
                                   reductionStageName, ttirOs);
      }

      int32_t kernelNumWarps = getKernelNumWarps(k, tritonNumWarps);
      int32_t kernelCudaThreadsPerCTA = kernelNumWarps * tritonThreadsPerWarp;

      emitJsonDescriptor(
          launchOp, k, blockX, blockY, blockZ, kernelId, emittedPtxIndex,
          kernelName, kernelNumWarps, tritonThreadsPerWarp, tritonNumStages,
          kernelCudaThreadsPerCTA, reductionStageId, jsonOs, firstKernel);

      ++emittedPtxIndex;

      if (isReduction) {
        emitJsonReductionStageDescriptor(
            k.elementType, k.reductionOperator, blockX, reductionStageId,
            emittedPtxIndex, reductionStageName, kernelNumWarps,
            tritonThreadsPerWarp, tritonNumStages, kernelCudaThreadsPerCTA,
            jsonOs, firstKernel);
        ++emittedPtxIndex;
      }

      ++fallbackId;
    });

    ttirOs << "}\n";

    jsonOs << "\n";
    jsonOs << "  ]\n";
    jsonOs << "}\n";

    if (failed) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCLowerToTritonPass() {
  return std::make_unique<FNACCLowerToTritonPass>();
}

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCLowerToTritonPass(
    llvm::StringRef ttirOutput, llvm::StringRef jsonOutput, int32_t numWarps,
    int32_t threadsPerWarp, int32_t numStages,
    llvm::StringRef f64MatmulStrategy) {
  return std::make_unique<FNACCLowerToTritonPass>(ttirOutput, jsonOutput,
                                                  numWarps, threadsPerWarp,
                                                  numStages, f64MatmulStrategy);
}

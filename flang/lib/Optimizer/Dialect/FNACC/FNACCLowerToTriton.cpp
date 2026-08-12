#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

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
  os << "  %tmp = arith.mulf %alpha_s, %av : tensor<" << block << "x" << elemTy
     << ">\n";
  os << "  %r = arith.addf %tmp, %bv : tensor<" << block << "x" << elemTy
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

    os << "  " << result << " = " << ttArithForExprKind(expr.kind) << " " << lhs
       << ", " << rhs << " : tensor<" << block << "x" << elemTy << ">\n";

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

static void emitTritonMatMul2DF32(const fir::fnacc::ElementwiseKernel &k,
                                  int64_t blockM, int64_t blockN,
                                  int64_t blockK, StringRef kernelName,
                                  llvm::raw_ostream &os) {
  assert(k.elementType == fir::fnacc::ElementType::F32 &&
         "matmul TTIR emitter is currently f32-only");

  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, "
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

  os << "  %zero = arith.constant 0.000000e+00 : f32\n";
  os << "  %acc0 = tt.splat %zero : f32 -> tensor<" << blockM << "x" << blockN
     << "xf32>\n";

  os << "  %c0_idx = arith.constant 0 : index\n";
  os << "  %bk_idx = arith.constant " << blockK << " : index\n";
  os << "  %k_idx = arith.index_cast %k : i32 to index\n";

  os << "  %acc = scf.for %kk_idx = %c0_idx to %k_idx step %bk_idx "
        "iter_args(%acc_body = %acc0) -> (tensor<"
     << blockM << "x" << blockN << "xf32>) {\n";

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

  os << "    %a_base = tt.splat %a : !tt.ptr<f32> -> tensor<" << blockM << "x"
     << blockK << "x!tt.ptr<f32>>\n";
  os << "    %b_base = tt.splat %b : !tt.ptr<f32> -> tensor<" << blockK << "x"
     << blockN << "x!tt.ptr<f32>>\n";
  os << "    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<" << blockM << "x"
     << blockK << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockK
     << "xi32>\n";
  os << "    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<" << blockK << "x"
     << blockN << "x!tt.ptr<f32>>, tensor<" << blockK << "x" << blockN
     << "xi32>\n";
  os << "    %a_tile = tt.load %a_ptrs, %mask_a : tensor<" << blockM << "x"
     << blockK << "x!tt.ptr<f32>>\n";
  os << "    %b_tile = tt.load %b_ptrs, %mask_b : tensor<" << blockK << "x"
     << blockN << "x!tt.ptr<f32>>\n";

  os << "    %acc_next = tt.dot %a_tile, %b_tile, %acc_body "
        "{inputPrecision = 0 : i32} : tensor<"
     << blockM << "x" << blockK << "xf32> * tensor<" << blockK << "x" << blockN
     << "xf32> -> tensor<" << blockM << "x" << blockN << "xf32>\n";
  os << "    scf.yield %acc_next : tensor<" << blockM << "x" << blockN
     << "xf32>\n";
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

  os << "  %c_base = tt.splat %c : !tt.ptr<f32> -> tensor<" << blockM << "x"
     << blockN << "x!tt.ptr<f32>>\n";
  os << "  %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<" << blockM << "x"
     << blockN << "x!tt.ptr<f32>>, tensor<" << blockM << "x" << blockN
     << "xi32>\n";
  os << "  tt.store %c_ptrs, %acc, %mask_c : tensor<" << blockM << "x" << blockN
     << "x!tt.ptr<f32>>\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonMatMul2DF64(const fir::fnacc::ElementwiseKernel &k,
                                  int64_t blockM, int64_t blockN,
                                  int64_t blockK, StringRef kernelName,
                                  llvm::raw_ostream &os) {
  // f64 matmul fallback:
  //
  //   C(i,j) = sum_p A(i,p) * B(p,j)
  //
  // This intentionally does not use tt.dot. Many Triton versions do not support
  // f64 dot in the same way as f32/f16/bf16/tf32 matmul. Instead, each program
  // computes a BLOCK_M x BLOCK_N C tile and loops over K one scalar position at
  // a time, broadcasting A(:,k) and B(k,:) to a tile.
  //
  // This is slower than the f32 tt.dot path, but it is correct and keeps the
  // ABI identical:
  //
  //   (%a, %b, %c, %n, %m, %k)

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

  os << "  %zero = arith.constant 0.000000e+00 : " << elemTy << "\n";

  os << "  %acc0 = tt.splat %zero : " << elemTy << " -> tensor<" << blockM
     << "x" << blockN << "x" << elemTy << ">\n";

  os << "  %c0_idx = arith.constant 0 : index\n";
  os << "  %c1_idx = arith.constant 1 : index\n";
  os << "  %k_idx = arith.index_cast %k : i32 to index\n";

  os << "  %acc = scf.for %kk_idx = %c0_idx to %k_idx step %c1_idx "
        "iter_args(%acc_body = %acc0) -> (tensor<"
     << blockM << "x" << blockN << "x" << elemTy << ">) {\n";

  os << "    %kk = arith.index_cast %kk_idx : index to i32\n";

  // A(i, kk) offset = i + kk * n
  os << "    %kk_m = tt.splat %kk : i32 -> tensor<" << blockM << "xi32>\n";
  os << "    %n_s_m_body = tt.splat %n : i32 -> tensor<" << blockM << "xi32>\n";
  os << "    %a_col = arith.muli %kk_m, %n_s_m_body : tensor<" << blockM
     << "xi32>\n";
  os << "    %a_offsets = arith.addi %offs_m, %a_col : tensor<" << blockM
     << "xi32>\n";

  // B(kk, j) offset = kk + j * k
  os << "    %kk_n = tt.splat %kk : i32 -> tensor<" << blockN << "xi32>\n";
  os << "    %k_s_n_body = tt.splat %k : i32 -> tensor<" << blockN << "xi32>\n";
  os << "    %b_col = arith.muli %offs_n, %k_s_n_body : tensor<" << blockN
     << "xi32>\n";
  os << "    %b_offsets = arith.addi %kk_n, %b_col : tensor<" << blockN
     << "xi32>\n";

  os << "    %a_base = tt.splat %a : " << ptrTy << " -> tensor<" << blockM
     << "x" << ptrTy << ">\n";
  os << "    %b_base = tt.splat %b : " << ptrTy << " -> tensor<" << blockN
     << "x" << ptrTy << ">\n";

  os << "    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<" << blockM << "x"
     << ptrTy << ">, tensor<" << blockM << "xi32>\n";
  os << "    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<" << blockN << "x"
     << ptrTy << ">, tensor<" << blockN << "xi32>\n";

  os << "    %a_vec = tt.load %a_ptrs, %mask_m : tensor<" << blockM << "x"
     << ptrTy << ">\n";
  os << "    %b_vec = tt.load %b_ptrs, %mask_n : tensor<" << blockN << "x"
     << ptrTy << ">\n";

  os << "    %a_e = tt.expand_dims %a_vec {axis = 1 : i32} : tensor<" << blockM
     << "x" << elemTy << "> -> tensor<" << blockM << "x1x" << elemTy << ">\n";

  os << "    %b_e = tt.expand_dims %b_vec {axis = 0 : i32} : tensor<" << blockN
     << "x" << elemTy << "> -> tensor<1x" << blockN << "x" << elemTy << ">\n";

  os << "    %a_b = tt.broadcast %a_e : tensor<" << blockM << "x1x" << elemTy
     << "> -> tensor<" << blockM << "x" << blockN << "x" << elemTy << ">\n";

  os << "    %b_b = tt.broadcast %b_e : tensor<1x" << blockN << "x" << elemTy
     << "> -> tensor<" << blockM << "x" << blockN << "x" << elemTy << ">\n";

  os << "    %prod = arith.mulf %a_b, %b_b : tensor<" << blockM << "x" << blockN
     << "x" << elemTy << ">\n";

  os << "    %acc_next = arith.addf %acc_body, %prod : tensor<" << blockM << "x"
     << blockN << "x" << elemTy << ">\n";

  os << "    scf.yield %acc_next : tensor<" << blockM << "x" << blockN << "x"
     << elemTy << ">\n";

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

static void emitJsonDescriptor(
    fir::fnacc::LaunchOp launchOp, const fir::fnacc::ElementwiseKernel &k,
    int64_t blockX, int64_t blockY, int64_t blockZ, int32_t kernelId,
    llvm::StringRef kernelName, int32_t tritonNumWarps,
    int32_t tritonThreadsPerWarp, int32_t tritonNumStages,
    int32_t cudaThreadsPerCTA, llvm::raw_ostream &os, bool &firstKernel) {
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

  std::string ptrJsonTy = jsonPtrType(k.elementType);
  std::string elemJsonTy = jsonElementType(k.elementType);

  os << "    {\n";
  os << "      \"id\": " << kernelId << ",\n";
  os << "      \"name\": \"" << kernelName << "\",\n";
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

  if (k.rank == 2) {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", "
       << "\"cdiv(extent_y, tile_y)\", \"1\"],\n";
  } else {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  }

  os << "      \"params\": [\n";

  unsigned nextSlot = 0;
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
  os << "      \"pack\": [";

  auto packVars = launchOp.getPackVars();
  llvm::ArrayRef<int32_t> targets = launchOp.getPackTargets();

  bool firstPack = true;
  for (auto it : llvm::enumerate(packVars)) {
    unsigned packIndex = it.index();
    Value packValue = it.value();

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

struct FNACCLowerToTritonPass
    : public fir::fnacc::impl::FNACCLowerToTritonBase<FNACCLowerToTritonPass> {
  FNACCLowerToTritonPass() = default;

  FNACCLowerToTritonPass(llvm::StringRef ttirOutput, llvm::StringRef jsonOutput,
                         int32_t numWarps, int32_t threadsPerWarp,
                         int32_t numStages) {
    this->ttirOutput = ttirOutput.str();
    this->jsonOutput = jsonOutput.str();
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numStages = numStages;
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

    bool hasSimpleElementwiseLaunch = false;
    bool hasMatmulLaunch = false;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);
      if (result.failed())
        return;

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();
      switch (k.kind) {
      case fir::fnacc::ElementwiseKernelKind::BinaryArrayArray:
      case fir::fnacc::ElementwiseKernelKind::Saxpy1D:
      case fir::fnacc::ElementwiseKernelKind::Expr1D:
      case fir::fnacc::ElementwiseKernelKind::Expr2D:
        hasSimpleElementwiseLaunch = true;
        break;
      case fir::fnacc::ElementwiseKernelKind::MatMul2D:
        hasMatmulLaunch = true;
        break;
      }
    });

    if (hasSimpleElementwiseLaunch && tritonNumWarps != 1) {
      if (hasMatmulLaunch) {
        module.emitWarning()
            << "FNACC module contains both simple elementwise and matmul "
               "kernels. TTIR currently has one module-wide num-warps setting, "
               "so simple elementwise lowering forces num-warps=1 for the "
               "whole module. Consider compiling matmul kernels separately.";
      } else {
        module.emitWarning()
            << "FNACC simple elementwise kernels currently use the single-warp "
               "Triton lowering path; using num-warps=1 instead of requested "
            << tritonNumWarps;
      }
      tritonNumWarps = 1;
    }

    int32_t cudaThreadsPerCTA = tritonNumWarps * tritonThreadsPerWarp;

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

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      if (failed)
        return;

      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);
      if (result.failed()) {
        launchOp.emitWarning("FNACC Triton emission skipped launch: ")
            << result.getFailure().reason;
        ++fallbackId;
        return;
      }

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();

      int32_t kernelId = getKernelId(launchOp, fallbackId);
      std::string kernelName = getKernelName(launchOp, kernelId);

      llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

      int64_t blockX = 1024;
      int64_t blockY = 1;
      int64_t blockZ = 1;

      if (k.rank == 2) {
        blockX = tiles.size() >= 1 ? tiles[0] : 16;
        blockY = tiles.size() >= 2 ? tiles[1] : 16;

        if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
          blockZ = tiles.size() >= 3 ? tiles[2] : 32;

          if (k.elementType == fir::fnacc::ElementType::F64) {
            emitTritonMatMul2DF64(k, blockX, blockY, blockZ, kernelName,
                                  ttirOs);
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

        if (k.kind == fir::fnacc::ElementwiseKernelKind::Expr1D)
          emitTritonExpr1D(k, blockX, kernelName, ttirOs);
        else if (k.kind == fir::fnacc::ElementwiseKernelKind::Saxpy1D)
          emitTritonSaxpy1D(k, blockX, kernelName, ttirOs);
        else
          emitTriton1D(k, blockX, kernelName, ttirOs);
      }

      emitJsonDescriptor(launchOp, k, blockX, blockY, blockZ, kernelId,
                         kernelName, tritonNumWarps, tritonThreadsPerWarp,
                         tritonNumStages, cudaThreadsPerCTA, jsonOs,
                         firstKernel);

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
    int32_t threadsPerWarp, int32_t numStages) {
  return std::make_unique<FNACCLowerToTritonPass>(
      ttirOutput, jsonOutput, numWarps, threadsPerWarp, numStages);
}

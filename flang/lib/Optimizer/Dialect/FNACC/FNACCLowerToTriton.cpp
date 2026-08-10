#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCLOWERTOTRITON
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

// Triton/NVVM-generated PTX currently appends two hidden pointer parameters
// after the explicit kernel parameters for the FNACC kernels we emit.
//
// Example 1-D binary PTX:
//
//   param_0 = a
//   param_1 = b
//   param_2 = c
//   param_3 = n
//   param_4 = hidden ptr
//   param_5 = hidden ptr
//
// The runtime appends two null CUdeviceptr values for these hidden args.
static constexpr int32_t kTritonHiddenPtrArgs = 2;

static constexpr llvm::StringLiteral kKernelIdAttrName = "fnacc.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fnacc.kernel_name";

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

static void emitTriton1D(const fir::fnacc::ElementwiseKernel &k, int64_t block,
                         StringRef kernelName, llvm::raw_ostream &os) {
  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %n: i32) "
     << "attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";

  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";

  os << "  %bS   = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %bS, %rng : tensor<" << block << "xi32>\n";

  os << "  %nS   = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %nS : tensor<" << block << "xi32>\n";

  auto loadArr = [&](StringRef ptr, StringRef dst) {
    os << "  " << dst << "p = tt.splat " << ptr << " : !tt.ptr<f32> -> tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  " << dst << "o = tt.addptr " << dst << "p, %offs : tensor<"
       << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

    os << "  " << dst << " = tt.load " << dst << "o, %mask : tensor<" << block
       << "x!tt.ptr<f32>>\n";
  };

  loadArr("%a", "%av");
  loadArr("%b", "%bv");

  os << "  %r = " << ttArith(k.computeOp) << " %av, %bv : tensor<" << block
     << "xf32>\n";

  os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<" << block
     << "x!tt.ptr<f32>>\n";

  os << "  %co = tt.addptr %cp, %offs : tensor<" << block
     << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

  os << "  tt.store %co, %r, %mask : tensor<" << block << "x!tt.ptr<f32>>\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonSaxpy1D(const fir::fnacc::ElementwiseKernel &k,
                              int64_t block, StringRef kernelName,
                              llvm::raw_ostream &os) {
  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, "
     << "%alpha: f32, %n: i32) attributes {noinline = false} {\n";

  os << "  %pid  = tt.get_program_id x : i32\n";
  os << "  %blk  = arith.constant " << block << " : i32\n";
  os << "  %base = arith.muli %pid, %blk : i32\n";

  os << "  %rng  = tt.make_range {start = 0 : i32, end = " << block
     << " : i32} : tensor<" << block << "xi32>\n";

  os << "  %bS   = tt.splat %base : i32 -> tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %bS, %rng : tensor<" << block << "xi32>\n";

  os << "  %nS   = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
  os << "  %mask = arith.cmpi slt, %offs, %nS : tensor<" << block << "xi32>\n";

  auto loadArr = [&](StringRef ptr, StringRef dst) {
    os << "  " << dst << "p = tt.splat " << ptr << " : !tt.ptr<f32> -> tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  " << dst << "o = tt.addptr " << dst << "p, %offs : tensor<"
       << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

    os << "  " << dst << " = tt.load " << dst << "o, %mask : tensor<" << block
       << "x!tt.ptr<f32>>\n";
  };

  loadArr("%a", "%av");
  loadArr("%b", "%bv");

  os << "  %alpha_s = tt.splat %alpha : f32 -> tensor<" << block << "xf32>\n";

  os << "  %tmp = arith.mulf %alpha_s, %av : tensor<" << block << "xf32>\n";
  os << "  %r = arith.addf %tmp, %bv : tensor<" << block << "xf32>\n";

  os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<" << block
     << "x!tt.ptr<f32>>\n";

  os << "  %co = tt.addptr %cp, %offs : tensor<" << block
     << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

  os << "  tt.store %co, %r, %mask : tensor<" << block << "x!tt.ptr<f32>>\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static int findValueIndex(ArrayRef<Value> values, Value value) {
  for (auto it : llvm::enumerate(values)) {
    if (it.value() == value)
      return static_cast<int>(it.index());
  }

  return -1;
}

struct ExprTritonEmitterState {
  int64_t block = 0;
  unsigned nextTmp = 0;

  llvm::SmallVector<bool> scalarSplatEmitted;
  llvm::SmallVector<std::string> scalarSplatNames;
};

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

static std::string emitExpr1D(const fir::fnacc::ElementwiseKernel &k,
                              const fir::fnacc::ElementwiseExpr &expr,
                              ExprTritonEmitterState &state,
                              llvm::raw_ostream &os) {
  int64_t block = state.block;

  switch (expr.kind) {
  case fir::fnacc::ElementwiseExprKind::ArrayLoad: {
    int index = findValueIndex(k.readArrays, expr.source);
    assert(index >= 0 && "array load source not found in read array list");

    if (index == 0)
      return "%read0v";
    if (index == 1)
      return "%read1v";

    llvm_unreachable("only two read arrays are currently supported");
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

      os << "  " << splatName << " = tt.splat " << scalarName
         << " : f32 -> tensor<" << block << "xf32>\n";

      state.scalarSplatEmitted[index] = true;
      state.scalarSplatNames[index] = splatName;
    }

    return state.scalarSplatNames[index];
  }

  case fir::fnacc::ElementwiseExprKind::ConstantF32: {
    unsigned id = state.nextTmp++;
    std::string cst = "%cst" + std::to_string(id);
    std::string splat = "%cst" + std::to_string(id) + "_s";

    os << "  " << cst << " = arith.constant "
       << static_cast<float>(expr.f32Value) << " : f32\n";

    os << "  " << splat << " = tt.splat " << cst << " : f32 -> tensor<" << block
       << "xf32>\n";

    return splat;
  }

  case fir::fnacc::ElementwiseExprKind::AddF:
  case fir::fnacc::ElementwiseExprKind::SubF:
  case fir::fnacc::ElementwiseExprKind::MulF:
  case fir::fnacc::ElementwiseExprKind::DivF: {
    assert(expr.operands.size() == 2 && "binary expression expected");

    std::string lhs = emitExpr1D(k, *expr.operands[0], state, os);
    std::string rhs = emitExpr1D(k, *expr.operands[1], state, os);

    std::string result = "%expr" + std::to_string(state.nextTmp++);

    os << "  " << result << " = " << ttArithForExprKind(expr.kind) << " " << lhs
       << ", " << rhs << " : tensor<" << block << "xf32>\n";

    return result;
  }
  }

  llvm_unreachable("unknown expression kind");
}

static void emitTritonExpr1D(const fir::fnacc::ElementwiseKernel &k,
                             int64_t block, StringRef kernelName,
                             llvm::raw_ostream &os) {
  assert(k.expression && "Expr1D kernel has no expression tree");
  assert(k.readArrays.size() == 2 &&
         "Expr1D TTIR emission currently requires exactly two read arrays");

  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>";

  for (unsigned i = 0; i < k.scalarRefs.size(); ++i)
    os << ", %scalar" << i << ": f32";

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

  auto loadArr = [&](StringRef ptr, StringRef dst) {
    os << "  " << dst << "p = tt.splat " << ptr << " : !tt.ptr<f32> -> tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  " << dst << "o = tt.addptr " << dst << "p, %offs : tensor<"
       << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

    os << "  " << dst << " = tt.load " << dst << "o, %mask : tensor<" << block
       << "x!tt.ptr<f32>>\n";
  };

  // For now the generic expression ABI still supports up to two read arrays.
  if (!k.readArrays.empty())
    loadArr("%a", "%read0v");

  if (k.readArrays.size() >= 2)
    loadArr("%b", "%read1v");

  ExprTritonEmitterState state;
  state.block = block;
  state.scalarSplatEmitted.resize(k.scalarRefs.size(), false);
  state.scalarSplatNames.resize(k.scalarRefs.size());

  std::string result = emitExpr1D(k, *k.expression, state, os);

  os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<" << block
     << "x!tt.ptr<f32>>\n";

  os << "  %co = tt.addptr %cp, %offs : tensor<" << block
     << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

  os << "  tt.store %co, " << result << ", %mask : tensor<" << block
     << "x!tt.ptr<f32>>\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTriton2D(const fir::fnacc::ElementwiseKernel &k, int64_t blockX,
                         int64_t blockY, StringRef kernelName,
                         llvm::raw_ostream &os) {
  int64_t block = blockX * blockY;

  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, "
     << "%n: i32, %m: i32) attributes {noinline = false} {\n";

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

  // Fortran column-major logical offset for zero-based generated coordinates:
  //
  //   offset = ix + jy * n
  //
  // This currently assumes supported loops are normal 1-based Fortran loops
  // and the generated GPU coordinates are zero-based offsets.
  os << "  %jy_n = arith.muli %jy, %n_s : tensor<" << block << "xi32>\n";
  os << "  %offs = arith.addi %ix, %jy_n : tensor<" << block << "xi32>\n";

  auto loadArr = [&](StringRef ptr, StringRef dst) {
    os << "  " << dst << "p = tt.splat " << ptr << " : !tt.ptr<f32> -> tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  " << dst << "o = tt.addptr " << dst << "p, %offs : tensor<"
       << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

    os << "  " << dst << " = tt.load " << dst << "o, %mask : tensor<" << block
       << "x!tt.ptr<f32>>\n";
  };

  loadArr("%a", "%av");
  loadArr("%b", "%bv");

  os << "  %rval = " << ttArith(k.computeOp) << " %av, %bv : tensor<" << block
     << "xf32>\n";

  os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<" << block
     << "x!tt.ptr<f32>>\n";

  os << "  %co = tt.addptr %cp, %offs : tensor<" << block
     << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

  os << "  tt.store %co, %rval, %mask : tensor<" << block << "x!tt.ptr<f32>>\n";

  os << "  tt.return\n";
  os << "}\n\n";
}

static void emitTritonMatMul2D(const fir::fnacc::ElementwiseKernel &k,
                               int64_t blockM, int64_t blockN, int64_t blockK,
                               StringRef kernelName, llvm::raw_ostream &os) {
  // Blocked matmul:
  //
  //   C(i,j) = sum_p A(i,p) * B(p,j)
  //
  // Column-major Fortran layout:
  //
  //   A(i,p) offset = i + p * n         A shape: n x k
  //   B(p,j) offset = p + j * k         B shape: k x m
  //   C(i,j) offset = i + j * n         C shape: n x m
  //
  // Program ids:
  //
  //   pid_m -> output row tile, i dimension
  //   pid_n -> output column tile, j dimension
  //
  // Each Triton program computes a BLOCK_M x BLOCK_N tile of C.

  os << "tt.func @" << kernelName
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, "
     << "%n: i32, %m: i32, %k: i32) attributes {noinline = false} {\n";

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

  // Build K offsets for this K block.
  os << "    %kk_s = tt.splat %kk_body : i32 -> tensor<" << blockK << "xi32>\n";
  os << "    %offs_k = arith.addi %kk_s, %offs_k0 : tensor<" << blockK
     << "xi32>\n";

  // A tile offsets: A(offs_m, offs_k)
  //
  // A offset = i + p * n
  //
  // Shapes:
  //   offs_m: BLOCK_M
  //   offs_k: BLOCK_K
  //
  // Want:
  //   a_offsets: BLOCK_M x BLOCK_K
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

  // B tile offsets: B(offs_k, offs_n)
  //
  // B offset = p + j * k
  //
  // Want:
  //   b_offsets: BLOCK_K x BLOCK_N
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

  // A mask: mask_m[:, None] & mask_k[None, :]
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

  // B mask: mask_k[:, None] & mask_n[None, :]
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

  // Pointer tensors.
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

  // Accumulate using dot.
  //
  // Depending on your Triton version, this may need slight syntax adjustment.
  os << "    %acc_next = tt.dot %a_tile, %b_tile, %acc_body "
        "{inputPrecision = 1 : i32} : tensor<"
     << blockM << "x" << blockK << "xf32> * tensor<" << blockK << "x" << blockN
     << "xf32> -> tensor<" << blockM << "x" << blockN << "xf32>\n";

  os << "    scf.yield %acc_next : tensor<" << blockM << "x" << blockN
     << "xf32>\n";

  os << "  }\n";

  // Store C tile.
  //
  // C offset = i + j * n
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

static llvm::SmallVector<unsigned>
kernelParamSlotsForValue(const fir::fnacc::ElementwiseKernel &k, Value v) {
  llvm::SmallVector<unsigned> slots;

  // Read arrays occupy slots:
  //
  //   read0 -> 0
  //   read1 -> 1
  //   read2 -> 2, if later supported
  for (unsigned i = 0; i < k.readArrays.size(); ++i) {
    if (k.readArrays[i] == v)
      slots.push_back(i);
  }

  // Write array occupies the slot immediately after read arrays.
  //
  // For current kernels:
  //
  //   read0 = slot 0
  //   read1 = slot 1
  //   write = slot 2
  //
  // If the same SSA value is both read and written, e.g.
  //
  //   c(i) = c(i) + b(i)
  //
  // then this intentionally adds both the read slot and the write slot.
  if (k.writeArray == v)
    slots.push_back(k.readArrays.size());

  // Scalars are after read arrays and write array.
  unsigned scalarBaseSlot = k.readArrays.size() + 1;
  for (unsigned i = 0; i < k.scalarRefs.size(); ++i) {
    if (k.scalarRefs[i] == v)
      slots.push_back(scalarBaseSlot + i);
  }

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
  else if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D)
    kindName = "matmul2d";

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

  if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
    os << "      \"grid\": [\"extent_x\", \"extent_y\", \"1\"],\n";
  } else if (k.rank == 2) {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", "
       << "\"cdiv(extent_y, tile_y)\", \"1\"],\n";
  } else {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  }

  os << "      \"params\": [\n";
  os << "        {\"slot\": 0, \"role\": \"read\",     \"name\": \"read0\",    "
        "\"type\": \"ptr<f32>\"},\n";
  os << "        {\"slot\": 1, \"role\": \"read\",     \"name\": \"read1\",    "
        "\"type\": \"ptr<f32>\"},\n";
  os << "        {\"slot\": 2, \"role\": \"write\",    \"name\": \"write\",    "
        "\"type\": \"ptr<f32>\"}";

  unsigned nextSlot = 3;

  for (unsigned i = 0; i < k.scalarRefs.size(); ++i) {
    os << ",\n";
    os << "        {\"slot\": " << nextSlot++
       << ", \"role\": \"scalar\",   \"name\": \"scalar" << i
       << "\",  \"type\": \"f32\"}";
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

    int32_t requestedNumWarps = this->numWarps;
    int32_t tritonNumWarps = requestedNumWarps;
    int32_t tritonThreadsPerWarp = this->threadsPerWarp;
    int32_t tritonNumStages = this->numStages;

    // The current FNACC elementwise TTIR path is intentionally vectorised
    // inside one Triton program and is known to lower reliably through the
    // single-warp TritonGPU pipeline. With num-warps > 1, some Triton versions
    // leave ttg.warp_id behind after LLVM lowering for these simple kernels.
    //
    // Matmul-style kernels may still benefit from multi-warp lowering, but
    // simple elementwise kernels are forced to one warp until the Triton
    // lowering pipeline is made version-robust for multi-warp elementwise
    // kernels.
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
               "kernels. "
               "TTIR currently has one module-wide num-warps setting, so "
               "simple "
               "elementwise lowering forces num-warps=1 for the whole module. "
               "Consider compiling matmul kernels separately.";
      } else {
        module.emitWarning()
            << "FNACC simple elementwise kernels currently use the single-warp "
               "Triton lowering path; using num-warps=1 instead of requested "
            << tritonNumWarps;
      }

      tritonNumWarps = 1;
    }

    int32_t cudaThreadsPerCTA = tritonNumWarps * tritonThreadsPerWarp;

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

    bool firstKernel = true;
    int32_t fallbackId = 0;

    ttirOs << "module attributes {"
           << "\"ttg.num-warps\" = " << tritonNumWarps << " : i32, "
           << "\"ttg.num-ctas\" = 1 : i32, "
           << "\"ttg.num-stages\" = " << tritonNumStages << " : i32, "
           << "\"ttg.threads-per-warp\" = " << tritonThreadsPerWarp << " : i32"
           << "} {\n";

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
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
          emitTritonMatMul2D(k, blockX, blockY, blockZ, kernelName, ttirOs);
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

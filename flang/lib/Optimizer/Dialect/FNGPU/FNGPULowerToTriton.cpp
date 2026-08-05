#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPULOWERTOTRITON
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

static constexpr int32_t kTritonNumWarps = 1;
static constexpr int32_t kTritonThreadsPerWarp = 32;
static constexpr int32_t kTritonNumCTAs = 1;
static constexpr int32_t kTritonNumStages = 3;

static constexpr int32_t kCudaThreadsPerCTA =
    kTritonNumWarps * kTritonThreadsPerWarp;

// Triton/NVVM-generated PTX currently appends two hidden pointer parameters
// after the explicit kernel parameters for the FNGPU kernels we emit.
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

static constexpr llvm::StringLiteral kKernelIdAttrName = "fngpu.kernel_id";
static constexpr llvm::StringLiteral kKernelNameAttrName = "fngpu.kernel_name";

static int32_t getKernelId(fir::fngpu::LaunchOp launchOp, int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName))
    return static_cast<int32_t>(attr.getInt());

  return fallbackId;
}

static std::string getKernelName(fir::fngpu::LaunchOp launchOp,
                                 int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<StringAttr>(kKernelNameAttrName))
    return attr.getValue().str();

  return "fngpu_kernel_" + std::to_string(fallbackId);
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

static void emitTriton1D(const fir::fngpu::ElementwiseKernel &k, int64_t block,
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

static void emitTritonSaxpy1D(const fir::fngpu::ElementwiseKernel &k,
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

static StringRef ttArithForExprKind(fir::fngpu::ElementwiseExprKind kind) {
  switch (kind) {
  case fir::fngpu::ElementwiseExprKind::AddF:
    return "arith.addf";
  case fir::fngpu::ElementwiseExprKind::SubF:
    return "arith.subf";
  case fir::fngpu::ElementwiseExprKind::MulF:
    return "arith.mulf";
  case fir::fngpu::ElementwiseExprKind::DivF:
    return "arith.divf";
  default:
    llvm_unreachable("not a binary arithmetic expression kind");
  }
}

static std::string emitExpr1D(const fir::fngpu::ElementwiseKernel &k,
                              const fir::fngpu::ElementwiseExpr &expr,
                              ExprTritonEmitterState &state,
                              llvm::raw_ostream &os) {
  int64_t block = state.block;

  switch (expr.kind) {
  case fir::fngpu::ElementwiseExprKind::ArrayLoad: {
    int index = findValueIndex(k.readArrays, expr.source);
    assert(index >= 0 && "array load source not found in read array list");

    if (index == 0)
      return "%read0v";
    if (index == 1)
      return "%read1v";

    llvm_unreachable("only two read arrays are currently supported");
  }

  case fir::fngpu::ElementwiseExprKind::ScalarLoad: {
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

  case fir::fngpu::ElementwiseExprKind::ConstantF32: {
    unsigned id = state.nextTmp++;
    std::string cst = "%cst" + std::to_string(id);
    std::string splat = "%cst" + std::to_string(id) + "_s";

    os << "  " << cst << " = arith.constant "
       << static_cast<float>(expr.f32Value) << " : f32\n";

    os << "  " << splat << " = tt.splat " << cst << " : f32 -> tensor<" << block
       << "xf32>\n";

    return splat;
  }

  case fir::fngpu::ElementwiseExprKind::AddF:
  case fir::fngpu::ElementwiseExprKind::SubF:
  case fir::fngpu::ElementwiseExprKind::MulF:
  case fir::fngpu::ElementwiseExprKind::DivF: {
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

static void emitTritonExpr1D(const fir::fngpu::ElementwiseKernel &k,
                             int64_t block, StringRef kernelName,
                             llvm::raw_ostream &os) {
  assert(k.expression && "Expr1D kernel has no expression tree");

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

static void emitTriton2D(const fir::fngpu::ElementwiseKernel &k, int64_t blockX,
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

static llvm::SmallVector<unsigned>
kernelParamSlotsForValue(const fir::fngpu::ElementwiseKernel &k, Value v) {
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

static void emitJsonDescriptor(fir::fngpu::LaunchOp launchOp,
                               const fir::fngpu::ElementwiseKernel &k,
                               int64_t blockX, int64_t blockY, int64_t blockZ,
                               int32_t kernelId, StringRef kernelName,
                               llvm::raw_ostream &os, bool &firstKernel) {
  if (!firstKernel)
    os << ",\n";
  firstKernel = false;

  StringRef kindName = "binary";
  if (k.kind == fir::fngpu::ElementwiseKernelKind::Saxpy1D)
    kindName = "saxpy1d";
  else if (k.kind == fir::fngpu::ElementwiseKernelKind::Expr1D)
    kindName = "expr1d";

  os << "    {\n";
  os << "      \"id\": " << kernelId << ",\n";
  os << "      \"name\": \"" << kernelName << "\",\n";
  os << "      \"kind\": \"" << kindName << "\",\n";
  os << "      \"rank\": " << k.rank << ",\n";
  os << "      \"tile\": [" << blockX << ", " << blockY << ", " << blockZ
     << "],\n";
  os << "      \"num_warps\": " << kTritonNumWarps << ",\n";
  os << "      \"threads_per_warp\": " << kTritonThreadsPerWarp << ",\n";
  os << "      \"num_ctas\": " << kTritonNumCTAs << ",\n";
  os << "      \"num_stages\": " << kTritonNumStages << ",\n";
  os << "      \"cuda_threads_per_cta\": " << kCudaThreadsPerCTA << ",\n";
  os << "      \"triton_hidden_ptr_args\": " << kTritonHiddenPtrArgs << ",\n";

  if (k.rank == 2) {
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
       << "\"type\": \"i32\"}\n";
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

struct FNGPULowerToTritonPass
    : public fir::fngpu::impl::FNGPULowerToTritonBase<FNGPULowerToTritonPass> {
  FNGPULowerToTritonPass() = default;

  FNGPULowerToTritonPass(llvm::StringRef ttirOutput,
                         llvm::StringRef jsonOutput) {
    this->ttirOutput = ttirOutput.str();
    this->jsonOutput = jsonOutput.str();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string ttirPath = this->ttirOutput;
    std::string jsonPath = this->jsonOutput;

    std::error_code ttirEc;
    llvm::raw_fd_ostream ttirOs(ttirPath, ttirEc);
    if (ttirEc) {
      module.emitError("cannot open FNGPU TTIR output file: ") << ttirPath;
      signalPassFailure();
      return;
    }

    std::error_code jsonEc;
    llvm::raw_fd_ostream jsonOs(jsonPath, jsonEc);
    if (jsonEc) {
      module.emitError("cannot open FNGPU JSON output file: ") << jsonPath;
      signalPassFailure();
      return;
    }

    jsonOs << "{\n";
    jsonOs << "  \"kernels\": [\n";

    bool firstKernel = true;
    int32_t fallbackId = 0;

    ttirOs << "module attributes {"
           << "\"ttg.num-warps\" = " << kTritonNumWarps << " : i32, "
           << "\"ttg.num-ctas\" = " << kTritonNumCTAs << " : i32, "
           << "\"ttg.num-stages\" = " << kTritonNumStages << " : i32, "
           << "\"ttg.threads-per-warp\" = " << kTritonThreadsPerWarp << " : i32"
           << "} {\n";

    module.walk([&](fir::fngpu::LaunchOp launchOp) {
      auto result = fir::fngpu::recognizeElementwiseKernel(launchOp);

      if (result.failed()) {
        launchOp.emitWarning("FNGPU Triton emission skipped launch: ")
            << result.getFailure().reason;
        ++fallbackId;
        return;
      }

      const fir::fngpu::ElementwiseKernel &k = result.getKernel();

      int32_t kernelId = getKernelId(launchOp, fallbackId);
      std::string kernelName = getKernelName(launchOp, kernelId);

      llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

      int64_t blockX = 1024;
      int64_t blockY = 1;
      int64_t blockZ = 1;

      if (k.rank == 2) {
        blockX = tiles.size() >= 1 ? tiles[0] : 16;
        blockY = tiles.size() >= 2 ? tiles[1] : 16;
        blockZ = 1;

        emitTriton2D(k, blockX, blockY, kernelName, ttirOs);
      } else {
        blockX = tiles.empty() ? 1024 : tiles[0];
        blockY = 1;
        blockZ = 1;

        if (k.kind == fir::fngpu::ElementwiseKernelKind::Expr1D)
          emitTritonExpr1D(k, blockX, kernelName, ttirOs);
        else if (k.kind == fir::fngpu::ElementwiseKernelKind::Saxpy1D)
          emitTritonSaxpy1D(k, blockX, kernelName, ttirOs);
        else
          emitTriton1D(k, blockX, kernelName, ttirOs);
      }

      emitJsonDescriptor(launchOp, k, blockX, blockY, blockZ, kernelId,
                         kernelName, jsonOs, firstKernel);

      ++fallbackId;
    });

    ttirOs << "}\n";

    jsonOs << "\n";
    jsonOs << "  ]\n";
    jsonOs << "}\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPULowerToTritonPass() {
  return std::make_unique<FNGPULowerToTritonPass>();
}

std::unique_ptr<mlir::Pass>
fir::fngpu::createFNGPULowerToTritonPass(llvm::StringRef ttirOutput,
                                         llvm::StringRef jsonOutput) {
  return std::make_unique<FNGPULowerToTritonPass>(ttirOutput, jsonOutput);
}

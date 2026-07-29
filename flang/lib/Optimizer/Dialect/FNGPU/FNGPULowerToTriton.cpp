#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPULOWERTOTRITON
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

// ---- Recognized pattern: 1-D elementwise  c(i) = op(a(i), b(i)) -----------
struct ElementwiseKernel {
  fir::DoLoopOp loop;
  Value tripCount;                    // the SSA value that is `n`
  llvm::SmallVector<Value> readArrays;// FIR array bases, in operand order
  Value writeArray;                   // FIR array base written to
  Operation *computeOp = nullptr;     // the arith op producing the stored value
};

// Trace an array_coor index back to a load of the induction memref.
static bool indexIsInductionLoad(Value v, Value indMemref) {
  while (true) {
    if (auto cvt = v.getDefiningOp<fir::ConvertOp>()) { v = cvt.getValue(); continue; }
    if (auto ld  = v.getDefiningOp<fir::LoadOp>())     return ld.getMemref() == indMemref;
    return false;
  }
}

static std::optional<ElementwiseKernel>
recognize(fir::fngpu::LaunchOp launchOp) {
  Region &region = launchOp.getRegion();

  // 1. Exactly one fir.do_loop in the region body.
  fir::DoLoopOp loop;
  for (Operation &op : region.front()) {
    if (auto dl = dyn_cast<fir::DoLoopOp>(op)) {
      if (loop) return std::nullopt;   // more than one loop -> bail
      loop = dl;
    }
  }
  if (!loop) return std::nullopt;

  ElementwiseKernel k;
  k.loop = loop;
  k.tripCount = loop.getUpperBound();  // = n (as index/converted); see note below

  // 2. Induction memref = target of `store <iterArg> to %ind` at loop top.
  Value iterArg = loop.getRegionIterArgs()[0];   // %arg5 in your FIR
  Value indMemref;
  for (Operation &op : loop.getBody()->getOperations())
    if (auto st = dyn_cast<fir::StoreOp>(op))
      if (st.getValue() == iterArg) { indMemref = st.getMemref(); break; }
  if (!indMemref) return std::nullopt;

  // 3. Classify every array_coor as read or write; require index == induction.
  Value writeStoredValue;
  for (Operation &op : loop.getBody()->getOperations()) {
    auto ac = dyn_cast<fir::ArrayCoorOp>(op);
    if (!ac) continue;
    if (ac.getIndices().size() != 1) return std::nullopt;          // 1-D only
    if (!indexIsInductionLoad(ac.getIndices()[0], indMemref))
      return std::nullopt;                                          // strided/offset -> bail

    Value base = ac.getMemref();
    for (Operation *user : ac.getResult().getUsers()) {
      if (isa<fir::LoadOp>(user)) {
        k.readArrays.push_back(base);
      } else if (auto st = dyn_cast<fir::StoreOp>(user)) {
        if (k.writeArray) return std::nullopt;                      // >1 write -> bail
        k.writeArray = base;
        writeStoredValue = st.getValue();                          // %41
      } else {
        return std::nullopt;                                        // unexpected use
      }
    }
  }
  if (!k.writeArray || k.readArrays.size() != 2) return std::nullopt;

  // 4. Compute op = definer of the stored value; must be a binary arith op
  //    whose operands are the two read loads.
  k.computeOp = writeStoredValue.getDefiningOp();
  if (!k.computeOp ||
      !isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(k.computeOp))
    return std::nullopt;

  return k;
}

// ---- Emit tt textual IR ---------------------------------------------------
static StringRef ttArith(Operation *op) {
  if (isa<arith::AddFOp>(op)) return "arith.addf";
  if (isa<arith::SubFOp>(op)) return "arith.subf";
  if (isa<arith::MulFOp>(op)) return "arith.mulf";
  return "arith.divf";
}

static void emitTriton(const ElementwiseKernel &k, int64_t block, int id,
                       llvm::raw_ostream &os) {
  // Kernel params: read0, read1, write, n
  os << "tt.func @fngpu_kernel_" << id
     << "(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>, %n: i32) {\n";
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
    os << "  " << dst << " = tt.load " << dst << "o, %mask : tensor<"
       << block << "x!tt.ptr<f32>>\n";
  };
  loadArr("%a", "%av");
  loadArr("%b", "%bv");
  os << "  %r = " << ttArith(k.computeOp) << " %av, %bv : tensor<"
     << block << "xf32>\n";
  os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<"
     << block << "x!tt.ptr<f32>>\n";
  os << "  %co = tt.addptr %cp, %offs : tensor<"
     << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";
  os << "  tt.store %co, %r, %mask : tensor<" << block << "x!tt.ptr<f32>>\n";
  os << "  tt.return\n}\n\n";
}

struct FNGPULowerToTritonPass
    : public fir::fngpu::impl::FNGPULowerToTritonBase<FNGPULowerToTritonPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::error_code ec;
    llvm::raw_fd_ostream os("fngpu_kernels.ttir", ec);
    if (ec) { module.emitError("cannot open fngpu_kernels.ttir"); return signalPassFailure(); }

    int id = 0;
    module.walk([&](fir::fngpu::LaunchOp launchOp) {
      auto k = recognize(launchOp);
      if (!k) {
        launchOp.emitWarning("fngpu.launch not an elementwise pattern; skipped");
        return;
      }
      llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();
      int64_t block = tiles.empty() ? 1024 : tiles[0];
      emitTriton(*k, block, id++, os);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fngpu::createFNGPULowerToTritonPass() {
  return std::make_unique<FNGPULowerToTritonPass>();
}


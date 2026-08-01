#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h"
#include "flang/Optimizer/Dialect/FNGPU/FNGPUDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace fir::fngpu {
#define GEN_PASS_DEF_FNGPULOWERTOTRITON
#include "flang/Optimizer/Dialect/FNGPU/FNGPUPasses.h.inc"
} // namespace fir::fngpu

using namespace mlir;

namespace {

  struct ElementwiseKernel {
    int rank = 1;

    fir::DoLoopOp loop1D;
    fir::DoLoopOp outerLoop;
    fir::DoLoopOp innerLoop;

    Value innerIndMemref; // i
    Value outerIndMemref; // j

    llvm::SmallVector<Value> readArrays;
    Value writeArray;

    Operation *computeOp = nullptr;
  };


  // Trace an array_coor index back to a load of the induction memref.
  static bool indexIsInductionLoad(Value v, Value indMemref) {
    while (true) {
      if (auto cvt = v.getDefiningOp<fir::ConvertOp>()) { v = cvt.getValue(); continue; }
      if (auto ld  = v.getDefiningOp<fir::LoadOp>())     return ld.getMemref() == indMemref;
      return false;
    }
  }

  static bool indexIsLoadOf(Value v, Value memref) {
    while (true) {
      if (auto cvt = v.getDefiningOp<fir::ConvertOp>()) {
	v = cvt.getValue();
	continue;
      }

      if (auto ld = v.getDefiningOp<fir::LoadOp>())
	return ld.getMemref() == memref;

      return false;
    }
  }

  static Value findInductionMemref(fir::DoLoopOp loop) {
    if (loop.getRegionIterArgs().empty())
      return {};

    Value iterArg = loop.getRegionIterArgs()[0];

    for (Operation &op : loop.getBody()->getOperations()) {
      if (auto st = dyn_cast<fir::StoreOp>(op)) {
	if (st.getValue() == iterArg)
	  return st.getMemref();
      }
    }

    return {};
  }

  static std::optional<ElementwiseKernel> recognize1D(fir::fngpu::LaunchOp launchOp) {
    Region &region = launchOp.getRegion();
    if (region.empty())
      return std::nullopt;

    Block &block = region.front();

    fir::DoLoopOp loop;
    for (Operation &op : block) {
      if (auto dl = dyn_cast<fir::DoLoopOp>(op)) {
	if (loop)
	  return std::nullopt;
	loop = dl;
      }
    }

    if (!loop)
      return std::nullopt;

    ElementwiseKernel k;
    k.rank = 1;
    k.loop1D = loop;

    Value indMemref = findInductionMemref(loop);
    if (!indMemref)
      return std::nullopt;

    k.innerIndMemref = indMemref;

    Value writeStoredValue;

    for (Operation &op : loop.getBody()->getOperations()) {
      auto ac = dyn_cast<fir::ArrayCoorOp>(op);
      if (!ac)
	continue;

      auto indices = ac.getIndices();
      if (indices.size() != 1)
	return std::nullopt;

      if (!indexIsLoadOf(indices[0], indMemref))
	return std::nullopt;

      Value base = ac.getMemref();

      for (Operation *user : ac.getResult().getUsers()) {
	if (isa<fir::LoadOp>(user)) {
	  k.readArrays.push_back(base);
	} else if (auto st = dyn_cast<fir::StoreOp>(user)) {
	  if (k.writeArray)
	    return std::nullopt;

	  k.writeArray = base;
	  writeStoredValue = st.getValue();
	} else {
	  return std::nullopt;
	}
      }
    }

    if (!k.writeArray)
      return std::nullopt;

    if (k.readArrays.size() != 2)
      return std::nullopt;

    k.computeOp = writeStoredValue.getDefiningOp();

    if (!k.computeOp ||
	!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(k.computeOp))
									 
      return std::nullopt;

    return k;
  }


  static std::optional<ElementwiseKernel> recognize2D(fir::fngpu::LaunchOp launchOp) {
    Region &region = launchOp.getRegion();
    if (region.empty())
      return std::nullopt;

    Block &launchBlock = region.front();

    // Launch region should contain exactly one outer loop.
    fir::DoLoopOp outer;
    for (Operation &op : launchBlock) {
      if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
	if (outer)
	  return std::nullopt;
	outer = loop;
      }
    }

    if (!outer)
      return std::nullopt;

    // Outer loop body should contain exactly one inner loop.
    fir::DoLoopOp inner;
    for (Operation &op : outer.getBody()->getOperations()) {
      if (auto loop = dyn_cast<fir::DoLoopOp>(op)) {
	if (inner)
	  return std::nullopt;
	inner = loop;
      }
    }

    if (!inner)
      return std::nullopt;

    ElementwiseKernel k;
    k.rank = 2;
    k.outerLoop = outer; // j
    k.innerLoop = inner; // i

    k.outerIndMemref = findInductionMemref(outer); // j alloca/declaration
    k.innerIndMemref = findInductionMemref(inner); // i alloca/declaration

    if (!k.outerIndMemref || !k.innerIndMemref)
      return std::nullopt;

    Value writeStoredValue;

    for (Operation &op : inner.getBody()->getOperations()) {
      auto ac = dyn_cast<fir::ArrayCoorOp>(op);
      if (!ac)
	continue;

      auto indices = ac.getIndices();
      if (indices.size() != 2)
	return std::nullopt;

      // FIR has array_coor %array %i, %j for a(i,j)
      if (!indexIsLoadOf(indices[0], k.innerIndMemref))
	return std::nullopt;

      if (!indexIsLoadOf(indices[1], k.outerIndMemref))
	return std::nullopt;

      Value base = ac.getMemref();

      for (Operation *user : ac.getResult().getUsers()) {
	if (isa<fir::LoadOp>(user)) {
	  k.readArrays.push_back(base);
	} else if (auto st = dyn_cast<fir::StoreOp>(user)) {
	  if (k.writeArray)
	    return std::nullopt;

	  k.writeArray = base;
	  writeStoredValue = st.getValue();
	} else {
	  return std::nullopt;
	}
      }
    }

    if (!k.writeArray)
      return std::nullopt;

    if (k.readArrays.size() != 2)
      return std::nullopt;

    k.computeOp = writeStoredValue.getDefiningOp();

    if (!k.computeOp ||
	!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(
									 k.computeOp))
      return std::nullopt;

    return k;
  }

  static std::optional<ElementwiseKernel> recognize(fir::fngpu::LaunchOp launchOp) {
    if (auto k2 = recognize2D(launchOp))
      return k2;

    return recognize1D(launchOp);
  }

  // ---- Emit tt textual IR ---------------------------------------------------
  static StringRef ttArith(Operation *op) {
    if (isa<arith::AddFOp>(op)) return "arith.addf";
    if (isa<arith::SubFOp>(op)) return "arith.subf";
    if (isa<arith::MulFOp>(op)) return "arith.mulf";
    return "arith.divf";
  }

  static void emitTriton1D(const ElementwiseKernel &k, int64_t block, int id,
			   llvm::raw_ostream &os) {
    // Kernel params: read0, read1, write, n
    os << "tt.func @fngpu_kernel_" << id
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

  static void emitTriton2D(const ElementwiseKernel &k,
			   int64_t blockX,
			   int64_t blockY,
			   int id,
			   llvm::raw_ostream &os) {
    int64_t block = blockX * blockY;

    os << "tt.func @fngpu_kernel_" << id
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

    os << "  %ix = arith.addi %base_x_s, %local_i : tensor<" << block << "xi32>\n";
    os << "  %jy = arith.addi %base_y_s, %local_j : tensor<" << block << "xi32>\n";

    os << "  %n_s = tt.splat %n : i32 -> tensor<" << block << "xi32>\n";
    os << "  %m_s = tt.splat %m : i32 -> tensor<" << block << "xi32>\n";

    os << "  %mask_x = arith.cmpi slt, %ix, %n_s : tensor<" << block << "xi32>\n";
    os << "  %mask_y = arith.cmpi slt, %jy, %m_s : tensor<" << block << "xi32>\n";
    os << "  %mask = arith.andi %mask_x, %mask_y : tensor<" << block << "xi1>\n";

    // Fortran column-major:
    // offset = ix + jy * n
    os << "  %jy_n = arith.muli %jy, %n_s : tensor<" << block << "xi32>\n";
    os << "  %offs = arith.addi %ix, %jy_n : tensor<" << block << "xi32>\n";

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

    os << "  %rval = " << ttArith(k.computeOp) << " %av, %bv : tensor<"
       << block << "xf32>\n";

    os << "  %cp = tt.splat %c : !tt.ptr<f32> -> tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  %co = tt.addptr %cp, %offs : tensor<"
       << block << "x!tt.ptr<f32>>, tensor<" << block << "xi32>\n";

    os << "  tt.store %co, %rval, %mask : tensor<"
       << block << "x!tt.ptr<f32>>\n";

    os << "  tt.return\n";
    os << "}\n\n";
  }


  static std::optional<unsigned>
  kernelParamSlotForValue(const ElementwiseKernel &k, Value v) {
    for (unsigned i = 0; i < k.readArrays.size(); ++i) {
      if (k.readArrays[i] == v)
	return i; // read0, read1, ...
    }

    if (k.writeArray == v)
      return k.readArrays.size(); // write slot, currently 2

    return std::nullopt;
  }

  static void emitJsonDescriptor(fir::fngpu::LaunchOp launchOp,
				 const ElementwiseKernel &k,
				 int64_t block,
				 int id,
				 llvm::raw_ostream &os,
				 bool &firstKernel) {
    if (!firstKernel)
      os << ",\n";
    firstKernel = false;

    os << "    {\n";
    os << "      \"name\": \"fngpu_kernel_" << id << "\",\n";
    os << "      \"block\": " << block << ",\n";
    os << "      \"grid\": \"cdiv(n, " << block << ")\",\n";

    os << "      \"params\": [\n";
    os << "        {\"slot\": 0, \"role\": \"read\",  \"name\": \"read0\", \"type\": \"ptr<f32>\"},\n";
    os << "        {\"slot\": 1, \"role\": \"read\",  \"name\": \"read1\", \"type\": \"ptr<f32>\"},\n";
    os << "        {\"slot\": 2, \"role\": \"write\", \"name\": \"write\", \"type\": \"ptr<f32>\"},\n";
    os << "        {\"slot\": 3, \"role\": \"size\",  \"name\": \"n\",     \"type\": \"i32\"}\n";
    os << "      ],\n";

    os << "      \"pack\": [";

    auto packVars = launchOp.getPackVars();
    llvm::ArrayRef<int32_t> targets = launchOp.getPackTargets();

    bool firstPack = true;
    for (auto it : llvm::enumerate(packVars)) {
      unsigned packIndex = it.index();
      Value packValue = it.value();

      std::optional<unsigned> slot = kernelParamSlotForValue(k, packValue);
      if (!slot) {
	launchOp.emitWarning(
			     "PACK variable was not used by recognized Triton kernel body");
	continue;
      }

      if (packIndex >= targets.size()) {
	launchOp.emitWarning("PACK target list shorter than PACK var list");
	continue;
      }

      int32_t target = targets[packIndex];

      if (!firstPack)
	os << ", ";
      firstPack = false;

      os << "{\"kernel_arg_slot\": " << *slot
	 << ", \"target\": " << target
	 << ", \"target_name\": \""
	 << (target == 0 ? "host" : "device")
	 << "\"}";
    }

    os << "]\n";
    os << "    }";
  }

  struct FNGPULowerToTritonPass
    : public fir::fngpu::impl::FNGPULowerToTritonBase<FNGPULowerToTritonPass> {
    void runOnOperation() override {
      ModuleOp module = getOperation();

      std::error_code ttirEc;
      llvm::raw_fd_ostream ttirOs("fngpu_kernels.ttir", ttirEc);
      if (ttirEc) {
	module.emitError("cannot open fngpu_kernels.ttir");
	signalPassFailure();
	return;
      }

      std::error_code jsonEc;
      llvm::raw_fd_ostream jsonOs("fngpu_kernels.json", jsonEc);
      if (jsonEc) {
	module.emitError("cannot open fngpu_kernels.json");
	signalPassFailure();
	return;
      }

      jsonOs << "{\n";
      jsonOs << "  \"kernels\": [\n";

      bool firstKernel = true;
      int id = 0;

  
      ttirOs << "module attributes {"
	     << "\"ttg.num-warps\" = 1 : i32, "
	     << "\"ttg.num-ctas\" = 1 : i32, "
	     << "\"ttg.num-stages\" = 3 : i32, "
	     << "\"ttg.threads-per-warp\" = 32 : i32"
	     << "} {\n";

      module.walk([&](fir::fngpu::LaunchOp launchOp) {
	auto k = recognize(launchOp);
	if (!k) {
	  launchOp.emitWarning("fngpu.launch not an elementwise pattern; skipped");
	  return;
	}

	llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

	int64_t jsonBlock = 1024;

	if (k->rank == 2) {
	  int64_t bx = tiles.size() >= 1 ? tiles[0] : 16;
	  int64_t by = tiles.size() >= 2 ? tiles[1] : 16;

	  emitTriton2D(*k, bx, by, id, ttirOs);

	  // For the old JSON scalar field, use total tile size for now.
	  // Later we should change JSON to store block: [bx, by].
	  jsonBlock = bx * by;
	} else {
	  int64_t block = tiles.empty() ? 1024 : tiles[0];

	  emitTriton1D(*k, block, id, ttirOs);

	  jsonBlock = block;
	}

	emitJsonDescriptor(launchOp, *k, jsonBlock, id, jsonOs, firstKernel);

	++id;

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


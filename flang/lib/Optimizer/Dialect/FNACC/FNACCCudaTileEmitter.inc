//===-- FNACCCudaTileEmitter.inc - Direct CUDA Tile emission ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Direct CUDA Tile IR emission. Included by the backend dispatcher so this
// optional text emitter adds no dependency on Triton or CUDA Tile C++
// libraries.
class CudaTileBackend final : public fir::fnacc::FNACCCodegenBackend {
  using Plan = fir::fnacc::FNACCKernelPlan;
  using Kind = fir::fnacc::ElementwiseKernelKind;
  using Expr = fir::fnacc::ElementwiseExpr;
  using EK = fir::fnacc::ElementwiseExprKind;
  using ET = fir::fnacc::ElementType;
  bool isHIP;

  static bool isSumReduction(const Plan &p) {
    return p.kernel.kind == Kind::ReductionSum1D ||
           p.kernel.kind == Kind::ReductionDot1D;
  }

  // One partial per tile. The same body, with a contiguous input, implements
  // the recursive finalization stage. Keep the neutral ABI parameter order.
  void emitReduction(const Plan &p, bool stage, llvm::raw_ostream &os) const {
    const auto &k = p.kernel;
    const auto &abi = stage ? p.reductionStage->abi : p.abi;
    std::string et = ttElementType(k.elementType).str();
    std::string shape = std::to_string(p.schedule.tile.x);
    auto ty = [&](const std::string &t) {
      return "tile<" + shape + "x" + t + ">";
    };
    auto splat = [&](const std::string &name, const std::string &value,
                     const std::string &type) {
      os << "    %" << name << "_r = reshape " << value << " : tile<" << type
         << "> -> tile<1x" << type << ">\n"
         << "    %" << name << " = broadcast %" << name << "_r : tile<1x"
         << type << "> -> " << ty(type) << "\n";
    };
    os << "  entry @" << (stage ? p.reductionStage->name : p.name) << "(";
    for (auto [i, a] : llvm::enumerate(abi.parameters)) {
      if (i)
        os << ", ";
      os << "%" << a.name << ": tile<";
      if (a.passing == fir::fnacc::FNACCKernelParameterPassing::DevicePointer)
        os << "ptr<";
      os << ttElementType(a.elementType);
      if (a.passing == fir::fnacc::FNACCKernelParameterPassing::DevicePointer)
        os << ">";
      os << ">";
    }
    os << ") {\n"
       << "    %pid_x, %pid_y, %pid_z = get_tile_block_id : tile<i32>\n"
       << "    %block = constant <i32: " << p.schedule.tile.x
       << "> : tile<i32>\n"
       << "    %base = muli %pid_x, %block : tile<i32>\n"
       << "    %range = iota : " << ty("i32") << "\n";
    splat("base_s", "%base", "i32");
    splat("extent_s", "%extent_x", "i32");
    os << "    %offs = addi %base_s, %range : " << ty("i32") << "\n"
       << "    %mask = cmpi less_than %offs, %extent_s, signed : " << ty("i32")
       << " -> " << ty("i1") << "\n"
       << "    %zero = constant <" << et << ": 0.0> : " << ty(et) << "\n"
       << "    %start = make_token : token\n";
    std::string token = "%start";
    unsigned reads = stage ? 1 : k.readArrays.size();
    for (unsigned i = 0; i < reads; ++i) {
      std::string stem = "r" + std::to_string(i);
      std::string offset = "%offs";
      if (!stage) {
        // Runtime checks the padded address range before launching, so these
        // signed i32 layout operations cannot overflow, even on masked lanes.
        os << "    %" << stem << "_delta = subi %loop_lower_x, %array" << i
           << "_lower0 : tile<i32>\n";
        splat(stem + "_delta_s", "%" + stem + "_delta", "i32");
        splat(stem + "_stride_s", "%array" + std::to_string(i) + "_stride0",
              "i32");
        os << "    %" << stem << "_logical = addi %offs, %" << stem
           << "_delta_s : " << ty("i32") << "\n"
           << "    %" << stem << "_offset = muli %" << stem << "_logical, %"
           << stem << "_stride_s : " << ty("i32") << "\n";
        offset = "%" + stem + "_offset";
      }
      splat(stem + "_ptr", stage ? "%input" : "%read" + std::to_string(i),
            "ptr<" + et + ">");
      os << "    %" << stem << "_addr = offset %" << stem << "_ptr, " << offset
         << " : " << ty("ptr<" + et + ">") << ", " << ty("i32") << " -> "
         << ty("ptr<" + et + ">") << "\n"
         << "    %" << stem << "_value, %" << stem
         << "_token = load_ptr_tko weak %" << stem
         << "_addr, %mask, %zero token=" << token << " : "
         << ty("ptr<" + et + ">") << ", " << ty("i1") << ", " << ty(et)
         << " -> " << ty(et) << ", token\n";
      token = "%" + stem + "_token";
    }
    std::string value = "%r0_value";
    if (!stage && k.kind == Kind::ReductionDot1D) {
      os << "    %product = mulf %r0_value, %r1_value : " << ty(et) << "\n";
      value = "%product";
    }
    os << "    %sum = reduce " << value << " dim=0 identities=[0.0 : " << et
       << "] : " << ty(et) << " -> tile<" << et << ">\n"
       << "      (%lhs: tile<" << et << ">, %rhs: tile<" << et << ">) {\n"
       << "        %combined = addf %lhs, %rhs : tile<" << et << ">\n"
       << "        yield %combined : tile<" << et << ">\n"
       << "      }\n"
       << "    %out = offset " << (stage ? "%output" : "%partials")
       << ", %pid_x : tile<ptr<" << et << ">>, tile<i32> -> tile<ptr<" << et
       << ">>\n"
       << "    %yes = constant <i1: 1> : tile<i1>\n"
       << "    %done = store_ptr_tko weak %out, %sum, %yes token=" << token
       << " : tile<ptr<" << et << ">>, tile<" << et << ">, tile<i1> -> token\n"
       << "    return\n  }\n";
  }

  static bool supportsExpr(const Expr &e, ET type) {
    if (e.elementType != ET::Unknown && e.elementType != type)
      return false;
    switch (e.kind) {
    case EK::ConstantReal:
      if (!std::isfinite(e.realValue))
        return false;
      break;
    case EK::ArrayLoad:
    case EK::ScalarLoad:
      break;
    case EK::AddF:
    case EK::SubF:
    case EK::MulF:
    case EK::SquareF:
      break;
    default:
      return false;
    }
    for (const auto &operand : e.operands)
      if (!supportsExpr(*operand, type))
        return false;
    return true;
  }

public:
  explicit CudaTileBackend(bool isHIP) : isHIP(isHIP) {}
  StringRef getName() const override { return "cuda-tile"; }
  StringRef getAcceleratorTarget() const override { return "cuda"; }
  fir::fnacc::FNACCDeviceIRKind getDeviceIRKind() const override {
    return fir::fnacc::FNACCDeviceIRKind::CUDATileIR;
  }
  fir::fnacc::FNACCDeviceImageKind getRuntimeImageKind() const override {
    return fir::fnacc::FNACCDeviceImageKind::Cubin;
  }
  int32_t getPrivatePointerArgumentCount(const Plan &) const override {
    return 0;
  }
  fir::fnacc::FNACCBackendSupport querySupport(const Plan &p) const override {
    using Support = fir::fnacc::FNACCBackendSupport;
    const auto &k = p.kernel;
    if (isHIP)
      return Support::failure("CUDA Tile requires a CUDA target");
    if (isSumReduction(p)) {
      if ((k.elementType != ET::F32 && k.elementType != ET::F64) ||
          k.rank != 1 || k.loopStepX != 1 || !k.indexRefs.empty() ||
          !k.scalarRefs.empty() || !k.outputs.empty() ||
          !k.reductionOutputs.empty() ||
          k.reductionOperator != fir::fnacc::ReductionOperator::Add ||
          !p.reductionStage || p.schedule.tile.x < 2 ||
          p.schedule.tile.x > 1024 ||
          (p.schedule.tile.x & (p.schedule.tile.x - 1)) != 0 ||
          k.arrayArguments.size() != k.readArrays.size())
        return Support::failure(
            "CUDA Tile sum/dot requires a rank-1 f32/f64 unit-step reduction "
            "and a power-of-two tile of 2..1024");
      if (k.kind == Kind::ReductionSum1D &&
          (k.readArrays.size() != 1 ||
           (k.expression && (k.expression->kind != EK::ArrayLoad ||
                             k.expression->source != k.readArrays[0]))))
        return Support::failure(
            "CUDA Tile sum currently requires a plain array value");
      if (k.kind == Kind::ReductionDot1D && k.readArrays.size() != 2)
        return Support::failure("CUDA Tile dot requires two input arrays");
      return Support::success();
    }
    if (k.kind != Kind::BinaryArrayArray && k.kind != Kind::Saxpy1D &&
        k.kind != Kind::Expr1D && k.kind != Kind::Expr2D)
      return Support::failure(
          "CUDA Tile currently supports simple pointwise arrays and rank-1 "
          "sum/dot; this kernel needs Triton");
    if ((k.elementType != ET::F32 && k.elementType != ET::F64) || k.rank < 1 ||
        k.rank > 2 || !k.outputs.empty() || !k.indexRefs.empty() ||
        k.loopStepX != 1 || k.loopStepY != 1)
      return Support::failure(
          "CUDA Tile requires homogeneous f32/f64 pointwise unit-step loops");
    auto powerOfTwo = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
    if (!powerOfTwo(p.schedule.tile.x) || !powerOfTwo(p.schedule.tile.y) ||
        p.schedule.tile.x > 1024 || p.schedule.tile.y > 1024 ||
        p.schedule.tile.x * p.schedule.tile.y > 1024)
      return Support::failure("CUDA Tile initial lowering requires "
                              "power-of-two tiles with at most 1024 elements");
    if (k.kind == Kind::BinaryArrayArray &&
        !isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(k.computeOp))
      return Support::failure(
          "CUDA Tile binary operation is not supported yet");
    if ((k.kind == Kind::Expr1D || k.kind == Kind::Expr2D) &&
        (!k.expression || !supportsExpr(*k.expression, k.elementType)))
      return Support::failure(
          "CUDA Tile expression needs an unsupported operation or conversion");
    return Support::success();
  }
  void beginModule(const fir::fnacc::FNACCKernelPlanOptions &,
                   llvm::raw_ostream &os) const override {
    os << "cuda_tile.module @fnacc_tiles {\n";
  }
  void endModule(llvm::raw_ostream &os) const override { os << "}\n"; }

  LogicalResult emitKernel(const Plan &p,
                           llvm::raw_ostream &os) const override {
    if (isSumReduction(p)) {
      emitReduction(p, false, os);
      emitReduction(p, true, os);
      return success();
    }
    const auto &k = p.kernel;
    std::string et = ttElementType(k.elementType).str();
    std::string shape = k.rank == 1 ? std::to_string(p.schedule.tile.x)
                                    : std::to_string(p.schedule.tile.y) + "x" +
                                          std::to_string(p.schedule.tile.x);
    auto ty = [&](const std::string &t) {
      return "tile<" + shape + "x" + t + ">";
    };
    unsigned next = 0;
    auto tmp = [&]() { return "%v" + std::to_string(next++); };
    auto splat = [&](const std::string &value, const std::string &type) {
      std::string r = tmp(), b = tmp();
      std::string ones = k.rank == 1 ? "1" : "1x1";
      os << "    " << r << " = reshape " << value << " : tile<" << type
         << "> -> tile<" << ones << "x" << type << ">\n";
      os << "    " << b << " = broadcast " << r << " : tile<" << ones << "x"
         << type << "> -> " << ty(type) << "\n";
      return b;
    };
    auto binary = [&](const std::string &op, const std::string &a,
                      const std::string &b, const std::string &type) {
      std::string r = tmp();
      os << "    " << r << " = " << op << " " << a << ", " << b << " : "
         << ty(type) << "\n";
      return r;
    };
    os << "  entry @" << p.name << "(";
    for (auto [i, a] : llvm::enumerate(p.abi.parameters)) {
      if (i)
        os << ", ";
      os << "%" << a.name << ": tile<";
      if (a.passing == fir::fnacc::FNACCKernelParameterPassing::DevicePointer)
        os << "ptr<";
      os << ttElementType(a.elementType);
      if (a.passing == fir::fnacc::FNACCKernelParameterPassing::DevicePointer)
        os << ">";
      os << ">";
    }
    os << ") {\n    %pid_x, %pid_y, %pid_z = get_tile_block_id : tile<i32>\n";
    auto coordinate = [&](const char *axis, int64_t block, bool x) {
      std::string c = tmp(), base = tmp(), range = tmp();
      os << "    " << c << " = constant <i32: " << block << "> : tile<i32>\n";
      os << "    " << base << " = muli %pid_" << axis << ", " << c
         << " : tile<i32>\n";
      os << "    " << range << " = iota : tile<" << block << "xi32>\n";
      if (k.rank == 2) {
        std::string r = tmp(), b = tmp();
        std::string rs =
            x ? "1x" + std::to_string(block) : std::to_string(block) + "x1";
        os << "    " << r << " = reshape " << range << " : tile<" << block
           << "xi32> -> tile<" << rs << "xi32>\n";
        os << "    " << b << " = broadcast " << r << " : tile<" << rs
           << "xi32> -> " << ty("i32") << "\n";
        range = b;
      }
      return binary("addi", splat(base, "i32"), range, "i32");
    };
    std::string x = coordinate("x", p.schedule.tile.x, true);
    std::string nx = splat("%extent_x", "i32"), mask = tmp();
    os << "    " << mask << " = cmpi less_than " << x << ", " << nx
       << ", signed : " << ty("i32") << " -> " << ty("i1") << "\n";
    std::string offset = x;
    if (k.rank == 2) {
      std::string y = coordinate("y", p.schedule.tile.y, false);
      std::string ny = splat("%extent_y", "i32"), my = tmp();
      os << "    " << my << " = cmpi less_than " << y << ", " << ny
         << ", signed : " << ty("i32") << " -> " << ty("i1") << "\n";
      mask = binary("andi", mask, my, "i1");
      offset = binary("addi", x, binary("muli", y, nx, "i32"), "i32");
    }
    std::vector<std::string> loads;
    std::string token = "%start";
    os << "    " << token << " = make_token : token\n";
    auto pointer = [&](const std::string &name) {
      std::string ptr = splat(name, "ptr<" + et + ">"), r = tmp();
      os << "    " << r << " = offset " << ptr << ", " << offset << " : "
         << ty("ptr<" + et + ">") << ", " << ty("i32") << " -> "
         << ty("ptr<" + et + ">") << "\n";
      return r;
    };
    for (unsigned i = 0; i < k.readArrays.size(); ++i) {
      std::string ptr = pointer("%read" + std::to_string(i)), v = tmp(),
                  t = tmp();
      os << "    " << v << ", " << t << " = load_ptr_tko weak " << ptr << ", "
         << mask << " token=" << token << " : " << ty("ptr<" + et + ">") << ", "
         << ty("i1") << " -> " << ty(et) << ", token\n";
      token = t;
      loads.push_back(v);
    }
    std::vector<std::string> scalars;
    for (unsigned i = 0; i < k.scalarRefs.size(); ++i)
      scalars.push_back(splat("%scalar" + std::to_string(i), et));
    std::function<std::string(const Expr &)> expression =
        [&](const Expr &e) -> std::string {
      if (e.kind == EK::ArrayLoad)
        return loads.at(findValueIndex(k.readArrays, e.source));
      if (e.kind == EK::ScalarLoad)
        return scalars.at(findValueIndex(k.scalarRefs, e.source));
      if (e.kind == EK::ConstantReal) {
        std::string r = tmp();
        std::ostringstream literal;
        literal << std::scientific << std::setprecision(17) << e.realValue;
        os << "    " << r << " = constant <" << et << ": " << literal.str()
           << "> : " << ty(et) << "\n";
        return r;
      }
      std::string a = expression(*e.operands[0]);
      std::string b = e.kind == EK::SquareF ? a : expression(*e.operands[1]);
      return binary(e.kind == EK::AddF   ? "addf"
                    : e.kind == EK::SubF ? "subf"
                                         : "mulf",
                    a, b, et);
    };
    std::string result;
    if (k.kind == Kind::Saxpy1D)
      result = binary("addf", binary("mulf", scalars.at(0), loads.at(0), et),
                      loads.at(1), et);
    else if (k.expression && (k.kind == Kind::Expr1D || k.kind == Kind::Expr2D))
      result = expression(*k.expression);
    else
      result = binary(isa<arith::AddFOp>(k.computeOp)   ? "addf"
                      : isa<arith::SubFOp>(k.computeOp) ? "subf"
                                                        : "mulf",
                      loads.at(0), loads.at(1), et);
    std::string out = pointer("%write");
    os << "    %done = store_ptr_tko weak " << out << ", " << result << ", "
       << mask << " token=" << token << " : " << ty("ptr<" + et + ">") << ", "
       << ty(et) << ", " << ty("i1") << " -> token\n    return\n  }\n";
    return success();
  }
};

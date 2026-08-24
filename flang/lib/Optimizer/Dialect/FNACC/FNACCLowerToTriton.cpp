#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelPlan.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCLOWERTOTRITON
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

static StringRef ttElementType(fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::I8:
    return "i8";
  case fir::fnacc::ElementType::I16:
    return "i16";
  case fir::fnacc::ElementType::I32:
    return "i32";
  case fir::fnacc::ElementType::I64:
    return "i64";
  case fir::fnacc::ElementType::F32:
    return "f32";
  case fir::fnacc::ElementType::F64:
    return "f64";
  default:
    llvm_unreachable("unsupported FNACC element type");
  }
}

static bool isIntegerElementType(fir::fnacc::ElementType type) {
  return type == fir::fnacc::ElementType::I8 ||
         type == fir::fnacc::ElementType::I16 ||
         type == fir::fnacc::ElementType::I32 ||
         type == fir::fnacc::ElementType::I64;
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

static bool isValidBackendName(StringRef name) {
  if (name.empty())
    return false;
  return llvm::all_of(name, [](char c) {
    return llvm::isAlnum(c) || c == '-' || c == '_' || c == '.' || c == '+';
  });
}

static StringRef deviceImageExtension(fir::fnacc::FNACCDeviceImageKind kind) {
  switch (kind) {
  case fir::fnacc::FNACCDeviceImageKind::PTX:
    return ".ptx";
  case fir::fnacc::FNACCDeviceImageKind::Cubin:
    return ".cubin";
  }
  llvm_unreachable("unknown FNACC device image kind");
}

static StringRef ttUnaryOpForExprKind(fir::fnacc::ElementwiseExprKind kind) {
  switch (kind) {
  case fir::fnacc::ElementwiseExprKind::NegF:
    return "arith.negf";
  case fir::fnacc::ElementwiseExprKind::AbsF:
    return "math.absf";
  case fir::fnacc::ElementwiseExprKind::SqrtF:
    return "math.sqrt";
  case fir::fnacc::ElementwiseExprKind::ExpF:
    return "math.exp";
  case fir::fnacc::ElementwiseExprKind::LogF:
    return "math.log";
  case fir::fnacc::ElementwiseExprKind::SinF:
    return "math.sin";
  case fir::fnacc::ElementwiseExprKind::CosF:
    return "math.cos";
  case fir::fnacc::ElementwiseExprKind::TanhF:
    return "math.tanh";
  case fir::fnacc::ElementwiseExprKind::AbsI:
    return "math.absi";
  default:
    llvm_unreachable("not a unary FNACC expression");
  }
}

static StringRef ttArith(Operation *op) {
  if (isa<arith::AddFOp>(op))
    return "arith.addf";
  if (isa<arith::SubFOp>(op))
    return "arith.subf";
  if (isa<arith::MulFOp>(op))
    return "arith.mulf";
  if (isa<arith::DivFOp>(op))
    return "arith.divf";
  if (isa<arith::AddIOp>(op))
    return "arith.addi";
  if (isa<arith::SubIOp>(op))
    return "arith.subi";
  if (isa<arith::MulIOp>(op))
    return "arith.muli";
  if (isa<arith::DivSIOp>(op))
    return "arith.divsi";
  llvm_unreachable("unsupported binary FNACC arithmetic operation");
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
  case fir::fnacc::ElementwiseExprKind::MinF:
    return "arith.minimumf";
  case fir::fnacc::ElementwiseExprKind::MaxF:
    return "arith.maximumf";
  case fir::fnacc::ElementwiseExprKind::MinNumF:
    return "arith.minnumf";
  case fir::fnacc::ElementwiseExprKind::MaxNumF:
    return "arith.maxnumf";
  case fir::fnacc::ElementwiseExprKind::AddI:
    return "arith.addi";
  case fir::fnacc::ElementwiseExprKind::SubI:
    return "arith.subi";
  case fir::fnacc::ElementwiseExprKind::MulI:
    return "arith.muli";
  case fir::fnacc::ElementwiseExprKind::DivSI:
    return "arith.divsi";
  case fir::fnacc::ElementwiseExprKind::MinSI:
    return "arith.minsi";
  case fir::fnacc::ElementwiseExprKind::MaxSI:
    return "arith.maxsi";
  default:
    llvm_unreachable("not a binary arithmetic expression kind");
  }
}

static StringRef comparisonPredicate(fir::fnacc::ElementwiseExprKind kind) {
  switch (kind) {
  case fir::fnacc::ElementwiseExprKind::CmpOLT:
    return "olt";
  case fir::fnacc::ElementwiseExprKind::CmpOLE:
    return "ole";
  case fir::fnacc::ElementwiseExprKind::CmpOGT:
    return "ogt";
  case fir::fnacc::ElementwiseExprKind::CmpOGE:
    return "oge";
  case fir::fnacc::ElementwiseExprKind::CmpOEQ:
    return "oeq";
  case fir::fnacc::ElementwiseExprKind::CmpONE:
    return "one";
  case fir::fnacc::ElementwiseExprKind::CmpSLT:
    return "slt";
  case fir::fnacc::ElementwiseExprKind::CmpSLE:
    return "sle";
  case fir::fnacc::ElementwiseExprKind::CmpSGT:
    return "sgt";
  case fir::fnacc::ElementwiseExprKind::CmpSGE:
    return "sge";
  case fir::fnacc::ElementwiseExprKind::CmpIEQ:
    return "eq";
  case fir::fnacc::ElementwiseExprKind::CmpINE:
    return "ne";
  default:
    llvm_unreachable("not a comparison");
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

    switch (index) {
    case 0:
      return "%read0v";
    case 1:
      return "%read1v";
    case 2:
      return "%read2v";
    default:
      llvm_unreachable("only three read arrays are supported");
    }
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
    std::string constant = "%cst" + std::to_string(id);
    std::string splat = "%cst" + std::to_string(id) + "_s";

    os << "  " << constant << " = arith.constant " << expr.realValue << " : "
       << elemTy << "\n";

    os << "  " << splat << " = tt.splat " << constant << " : " << elemTy
       << " -> tensor<" << block << "x" << elemTy << ">\n";

    return splat;
  }

  case fir::fnacc::ElementwiseExprKind::ConstantInteger: {
    unsigned id = state.nextTmp++;
    std::string constant = "%cst" + std::to_string(id);
    std::string splat = "%cst" + std::to_string(id) + "_s";

    os << "  " << constant << " = arith.constant " << expr.integerValue << " : "
       << elemTy << "\n";
    os << "  " << splat << " = tt.splat " << constant << " : " << elemTy
       << " -> tensor<" << block << "x" << elemTy << ">\n";
    return splat;
  }

  case fir::fnacc::ElementwiseExprKind::NegF:
  case fir::fnacc::ElementwiseExprKind::AbsF:
  case fir::fnacc::ElementwiseExprKind::SqrtF:
  case fir::fnacc::ElementwiseExprKind::ExpF:
  case fir::fnacc::ElementwiseExprKind::LogF:
  case fir::fnacc::ElementwiseExprKind::SinF:
  case fir::fnacc::ElementwiseExprKind::CosF:
  case fir::fnacc::ElementwiseExprKind::TanhF: {
    assert(expr.operands.size() == 1 &&
           "unary expression requires one operand");

    std::string operand = emitExprVector(k, *expr.operands[0], state, os);
    std::string result = "%expr" + std::to_string(state.nextTmp++);

    os << "  " << result << " = " << ttUnaryOpForExprKind(expr.kind) << " "
       << operand << " : tensor<" << block << "x" << elemTy << ">\n";

    return result;
  }

  case fir::fnacc::ElementwiseExprKind::AbsI: {
    assert(expr.operands.size() == 1 &&
           "integer absolute value requires one operand");

    std::string operand = emitExprVector(k, *expr.operands[0], state, os);
    std::string zero = "%cst" + std::to_string(state.nextTmp++);
    std::string zeroSplat = zero + "_s";
    std::string negative = "%expr" + std::to_string(state.nextTmp++);
    std::string predicate = "%pred" + std::to_string(state.nextTmp++);
    std::string result = "%expr" + std::to_string(state.nextTmp++);
    os << "  " << zero << " = arith.constant 0 : " << elemTy << "\n";
    os << "  " << zeroSplat << " = tt.splat " << zero << " : " << elemTy
       << " -> tensor<" << block << "x" << elemTy << ">\n";
    os << "  " << negative << " = arith.subi " << zeroSplat << ", " << operand
       << " : tensor<" << block << "x" << elemTy << ">\n";
    os << "  " << predicate << " = arith.cmpi slt, " << operand << ", "
       << zeroSplat << " : tensor<" << block << "x" << elemTy << ">\n";
    os << "  " << result << " = arith.select " << predicate << ", " << negative
       << ", " << operand << " : tensor<" << block << "xi1>, tensor<" << block
       << "x" << elemTy << ">\n";
    return result;
  }

  case fir::fnacc::ElementwiseExprKind::AddF:
  case fir::fnacc::ElementwiseExprKind::SubF:
  case fir::fnacc::ElementwiseExprKind::MulF:
  case fir::fnacc::ElementwiseExprKind::DivF:
  case fir::fnacc::ElementwiseExprKind::MinF:
  case fir::fnacc::ElementwiseExprKind::MaxF:
  case fir::fnacc::ElementwiseExprKind::MinNumF:
  case fir::fnacc::ElementwiseExprKind::MaxNumF: {
    assert(expr.operands.size() == 2 &&
           "binary expression requires two operands");

    std::string lhs = emitExprVector(k, *expr.operands[0], state, os);
    std::string rhs = emitExprVector(k, *expr.operands[1], state, os);
    std::string result = "%expr" + std::to_string(state.nextTmp++);

    os << "  " << result << " = " << ttArithForExprKind(expr.kind) << " " << lhs
       << ", " << rhs << " : tensor<" << block << "x" << elemTy << ">\n";

    return result;
  }

  case fir::fnacc::ElementwiseExprKind::AddI:
  case fir::fnacc::ElementwiseExprKind::SubI:
  case fir::fnacc::ElementwiseExprKind::MulI:
  case fir::fnacc::ElementwiseExprKind::DivSI:
  case fir::fnacc::ElementwiseExprKind::MinSI:
  case fir::fnacc::ElementwiseExprKind::MaxSI: {
    assert(expr.operands.size() == 2 &&
           "binary integer expression requires two operands");

    std::string lhs = emitExprVector(k, *expr.operands[0], state, os);
    std::string rhs = emitExprVector(k, *expr.operands[1], state, os);
    std::string result = "%expr" + std::to_string(state.nextTmp++);

    os << "  " << result << " = " << ttArithForExprKind(expr.kind) << " " << lhs
       << ", " << rhs << " : tensor<" << block << "x" << elemTy << ">\n";
    return result;
  }

  case fir::fnacc::ElementwiseExprKind::CmpOLT:
  case fir::fnacc::ElementwiseExprKind::CmpOLE:
  case fir::fnacc::ElementwiseExprKind::CmpOGT:
  case fir::fnacc::ElementwiseExprKind::CmpOGE:
  case fir::fnacc::ElementwiseExprKind::CmpOEQ:
  case fir::fnacc::ElementwiseExprKind::CmpONE: {
    assert(expr.operands.size() == 2 && "comparison requires two operands");

    std::string lhs = emitExprVector(k, *expr.operands[0], state, os);
    std::string rhs = emitExprVector(k, *expr.operands[1], state, os);
    std::string result = "%pred" + std::to_string(state.nextTmp++);

    os << "  " << result << " = arith.cmpf " << comparisonPredicate(expr.kind)
       << ", " << lhs << ", " << rhs << " : tensor<" << block << "x" << elemTy
       << ">\n";

    return result;
  }

  case fir::fnacc::ElementwiseExprKind::CmpSLT:
  case fir::fnacc::ElementwiseExprKind::CmpSLE:
  case fir::fnacc::ElementwiseExprKind::CmpSGT:
  case fir::fnacc::ElementwiseExprKind::CmpSGE:
  case fir::fnacc::ElementwiseExprKind::CmpIEQ:
  case fir::fnacc::ElementwiseExprKind::CmpINE: {
    assert(expr.operands.size() == 2 &&
           "integer comparison requires two operands");
    std::string lhs = emitExprVector(k, *expr.operands[0], state, os);
    std::string rhs = emitExprVector(k, *expr.operands[1], state, os);
    std::string result = "%pred" + std::to_string(state.nextTmp++);

    os << "  " << result << " = arith.cmpi " << comparisonPredicate(expr.kind)
       << ", " << lhs << ", " << rhs << " : tensor<" << block << "x" << elemTy
       << ">\n";
    return result;
  }

  case fir::fnacc::ElementwiseExprKind::Select: {
    assert(expr.operands.size() == 3 &&
           "select requires condition, true value, and false value");

    std::string condition = emitExprVector(k, *expr.operands[0], state, os);
    std::string trueValue = emitExprVector(k, *expr.operands[1], state, os);
    std::string falseValue = emitExprVector(k, *expr.operands[2], state, os);
    std::string result = "%expr" + std::to_string(state.nextTmp++);

    os << "  " << result << " = arith.select " << condition << ", " << trueValue
       << ", " << falseValue << " : tensor<" << block << "xi1>, "
       << "tensor<" << block << "x" << elemTy << ">\n";

    return result;
  }

  default:
    llvm_unreachable("unsupported FNACC expression reached TTIR emission");
  }
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

static StringRef reductionArithOp(fir::fnacc::ReductionOperator op,
                                  fir::fnacc::ElementType type) {
  bool integer = isIntegerElementType(type);
  switch (op) {
  case fir::fnacc::ReductionOperator::Add:
    return integer ? "arith.addi" : "arith.addf";
  case fir::fnacc::ReductionOperator::Multiply:
    return integer ? "arith.muli" : "arith.mulf";
  case fir::fnacc::ReductionOperator::Min:
    return integer ? "arith.minsi" : "arith.minimumf";
  case fir::fnacc::ReductionOperator::Max:
    return integer ? "arith.maxsi" : "arith.maximumf";
  }
  llvm_unreachable("unknown FNACC reduction operator");
}

static StringRef reductionIdentity(fir::fnacc::ReductionOperator op,
                                   fir::fnacc::ElementType type) {
  switch (op) {
  case fir::fnacc::ReductionOperator::Add:
    return isIntegerElementType(type) ? "0" : "0.000000e+00";
  case fir::fnacc::ReductionOperator::Multiply:
    return isIntegerElementType(type) ? "1" : "1.000000e+00";
  case fir::fnacc::ReductionOperator::Min:
    switch (type) {
    case fir::fnacc::ElementType::I8:
      return "127";
    case fir::fnacc::ElementType::I16:
      return "32767";
    case fir::fnacc::ElementType::I32:
      return "2147483647";
    case fir::fnacc::ElementType::I64:
      return "9223372036854775807";
    case fir::fnacc::ElementType::F32:
      return "0x7F800000";
    case fir::fnacc::ElementType::F64:
      return "0x7FF0000000000000";
    case fir::fnacc::ElementType::Unknown:
      break;
    }
    break;
  case fir::fnacc::ReductionOperator::Max:
    switch (type) {
    case fir::fnacc::ElementType::I8:
      return "-128";
    case fir::fnacc::ElementType::I16:
      return "-32768";
    case fir::fnacc::ElementType::I32:
      return "-2147483648";
    case fir::fnacc::ElementType::I64:
      return "-9223372036854775808";
    case fir::fnacc::ElementType::F32:
      return "0xFF800000";
    case fir::fnacc::ElementType::F64:
      return "0xFFF0000000000000";
    case fir::fnacc::ElementType::Unknown:
      break;
    }
    break;
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
  os << "    %r = " << reductionArithOp(k.reductionOperator, k.elementType)
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
  os << "    %r = " << reductionArithOp(op, type) << " %lhs, %rhs : " << elemTy
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

  os << "  %prod = "
     << (isIntegerElementType(k.elementType) ? "arith.muli" : "arith.mulf")
     << " %av, %bv : tensor<" << block << "x" << elemTy << ">\n";

  os << "  %zero = arith.constant "
     << (isIntegerElementType(k.elementType) ? "0" : "0.000000e+00") << " : "
     << elemTy << "\n";
  os << "  %zero_s = tt.splat %zero : " << elemTy << " -> tensor<" << block
     << "x" << elemTy << ">\n";

  os << "  %safe = arith.select %mask, %prod, %zero_s : tensor<" << block
     << "xi1>, tensor<" << block << "x" << elemTy << ">\n";

  os << "  %sum = \"tt.reduce\"(%safe) ({\n";
  os << "  ^bb0(%lhs: " << elemTy << ", %rhs: " << elemTy << "):\n";
  os << "    %r = "
     << (isIntegerElementType(k.elementType) ? "arith.addi" : "arith.addf")
     << " %lhs, %rhs : " << elemTy << "\n";
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

static StringRef jsonParameterRole(fir::fnacc::FNACCKernelParameterRole role) {
  using Role = fir::fnacc::FNACCKernelParameterRole;
  switch (role) {
  case Role::Read:
    return "read";
  case Role::Write:
    return "write";
  case Role::Partials:
    return "partials";
  case Role::Scalar:
    return "scalar";
  case Role::ExtentX:
    return "extent_x";
  case Role::ExtentY:
    return "extent_y";
  case Role::ExtentZ:
    return "extent_k";
  }
  llvm_unreachable("unknown FNACC ABI parameter role");
}

static std::string
jsonParameterType(const fir::fnacc::FNACCKernelParameter &parameter) {
  if (parameter.passing ==
      fir::fnacc::FNACCKernelParameterPassing::DevicePointer)
    return jsonPtrType(parameter.elementType);
  return jsonElementType(parameter.elementType);
}

static void emitJsonABI(const fir::fnacc::FNACCKernelABI &abi,
                        llvm::raw_ostream &os) {
  os << "      \"params\": [\n";
  for (auto [index, parameter] : llvm::enumerate(abi.parameters)) {
    if (index != 0)
      os << ",\n";
    os << "        {\"slot\": " << parameter.slot << ", \"role\": \""
       << jsonParameterRole(parameter.role) << "\", \"name\": \""
       << parameter.name << "\", \"type\": \"" << jsonParameterType(parameter)
       << "\"}";
  }
  os << "\n      ],\n";

  os << "      \"pack\": [";
  for (auto [index, binding] : llvm::enumerate(abi.packBindings)) {
    if (index != 0)
      os << ", ";
    os << "{\"kernel_arg_slot\": " << binding.kernelArgSlot
       << ", \"target\": " << binding.target << ", \"target_name\": \""
       << (binding.target == 0 ? "host" : "device") << "\"}";
  }
  os << "]\n";
}

static void emitJsonDescriptor(const fir::fnacc::FNACCKernelPlan &plan,
                               int32_t ptxIndex,
                               const fir::fnacc::FNACCCodegenBackend &backend,
                               llvm::raw_ostream &os, bool &firstKernel) {
  const fir::fnacc::ElementwiseKernel &k = plan.kernel;
  const fir::fnacc::FNACCKernelSchedule &schedule = plan.schedule;

  if (!firstKernel)
    os << ",\n";
  firstKernel = false;

  os << "    {\n";
  os << "      \"id\": " << plan.id << ",\n";
  os << "      \"name\": \"" << plan.name << "\",\n";
  os << "      \"backend\": \"" << backend.getName() << "\",\n";
  os << "      \"device_ir_kind\": \""
     << fir::fnacc::fnaccDeviceIRKindName(backend.getDeviceIRKind()) << "\",\n";
  os << "      \"device_image_kind\": \""
     << fir::fnacc::fnaccDeviceImageKindName(backend.getRuntimeImageKind())
     << "\",\n";
  os << "      \"image_index\": " << ptxIndex << ",\n";
  os << "      \"image_file\": \"" << plan.name
     << deviceImageExtension(backend.getRuntimeImageKind()) << "\",\n";
  // Legacy aliases consumed by existing wrappers and runtimes.
  os << "      \"ptx_index\": " << ptxIndex << ",\n";
  os << "      \"ptx_file\": \"" << plan.name << ".ptx\",\n";
  os << "      \"kind\": \"" << fir::fnacc::fnaccKernelKindName(k.kind)
     << "\",\n";
  os << "      \"rank\": " << k.rank << ",\n";
  os << "      \"tile\": [" << schedule.tile.x << ", " << schedule.tile.y
     << ", " << schedule.tile.z << "],\n";
  os << "      \"num_warps\": " << schedule.parallelSubgroups << ",\n";
  os << "      \"threads_per_warp\": " << schedule.subgroupWidth << ",\n";
  os << "      \"num_ctas\": 1,\n";
  os << "      \"num_stages\": " << schedule.pipelineStages << ",\n";
  os << "      \"cuda_threads_per_cta\": "
     << schedule.parallelSubgroups * schedule.subgroupWidth << ",\n";
  os << "      \"private_pointer_args\": "
     << backend.getPrivatePointerArgumentCount(plan) << ",\n";
  os << "      \"triton_hidden_ptr_args\": "
     << backend.getPrivatePointerArgumentCount(plan) << ",\n";

  if (plan.reductionStage)
    os << "      \"reduction_stage_id\": " << plan.reductionStage->id << ",\n";

  if (k.rank == 2) {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", "
       << "\"cdiv(extent_y, tile_y)\", \"1\"],\n";
  } else {
    os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  }

  if (fir::fnacc::isReductionKernelKind(k.kind))
    os << "      \"reduction_op\": \""
       << reductionOperatorName(k.reductionOperator) << "\",\n";

  emitJsonABI(plan.abi, os);
  os << "    }";
}

static void
emitJsonReductionStageDescriptor(const fir::fnacc::FNACCKernelPlan &plan,
                                 int32_t ptxIndex,
                                 const fir::fnacc::FNACCCodegenBackend &backend,
                                 llvm::raw_ostream &os, bool &firstKernel) {
  assert(plan.reductionStage && "reduction stage descriptor without a stage");
  const fir::fnacc::FNACCReductionStagePlan &stage = *plan.reductionStage;

  if (!firstKernel)
    os << ",\n";
  firstKernel = false;

  os << "    {\n";
  os << "      \"id\": " << stage.id << ",\n";
  os << "      \"name\": \"" << stage.name << "\",\n";
  os << "      \"backend\": \"" << backend.getName() << "\",\n";
  os << "      \"device_ir_kind\": \""
     << fir::fnacc::fnaccDeviceIRKindName(backend.getDeviceIRKind()) << "\",\n";
  os << "      \"device_image_kind\": \""
     << fir::fnacc::fnaccDeviceImageKindName(backend.getRuntimeImageKind())
     << "\",\n";
  os << "      \"image_index\": " << ptxIndex << ",\n";
  os << "      \"image_file\": \"" << stage.name
     << deviceImageExtension(backend.getRuntimeImageKind()) << "\",\n";
  // Legacy aliases consumed by existing wrappers and runtimes.
  os << "      \"ptx_index\": " << ptxIndex << ",\n";
  os << "      \"ptx_file\": \"" << stage.name << ".ptx\",\n";
  os << "      \"kind\": \"reduction_stage1d\",\n";
  os << "      \"reduction_op\": \""
     << reductionOperatorName(stage.reductionOperator) << "\",\n";
  os << "      \"rank\": 1,\n";
  os << "      \"tile\": [" << plan.schedule.tile.x << ", 1, 1],\n";
  os << "      \"num_warps\": " << plan.schedule.parallelSubgroups << ",\n";
  os << "      \"threads_per_warp\": " << plan.schedule.subgroupWidth << ",\n";
  os << "      \"num_ctas\": 1,\n";
  os << "      \"num_stages\": " << plan.schedule.pipelineStages << ",\n";
  os << "      \"cuda_threads_per_cta\": "
     << plan.schedule.parallelSubgroups * plan.schedule.subgroupWidth << ",\n";
  os << "      \"triton_hidden_ptr_args\": "
     << backend.getPrivatePointerArgumentCount(plan) << ",\n";
  os << "      \"grid\": [\"cdiv(extent_x, tile_x)\", \"1\", \"1\"],\n";
  emitJsonABI(stage.abi, os);
  os << "    }";
}

class TritonBackend final : public fir::fnacc::FNACCCodegenBackend {
public:
  StringRef getName() const override { return "triton"; }

  fir::fnacc::FNACCDeviceImageKind getRuntimeImageKind() const override {
    return fir::fnacc::FNACCDeviceImageKind::PTX;
  }

  fir::fnacc::FNACCBackendSupport
  querySupport(const fir::fnacc::FNACCKernelPlan &plan) const override {
    const fir::fnacc::ElementwiseKernel &kernel = plan.kernel;
    const fir::fnacc::FNACCKernelSchedule &schedule = plan.schedule;

    if (kernel.rank < 1 || kernel.rank > 2)
      return fir::fnacc::FNACCBackendSupport::failure(
          "Triton backend supports rank-one and rank-two kernels");

    if (kernel.elementType == fir::fnacc::ElementType::Unknown)
      return fir::fnacc::FNACCBackendSupport::failure(
          "kernel has no supported element type");

    if (schedule.tile.x <= 0 || schedule.tile.y <= 0 || schedule.tile.z <= 0)
      return fir::fnacc::FNACCBackendSupport::failure(
          "tile dimensions must be positive");

    if (schedule.parallelSubgroups <= 0 || schedule.pipelineStages <= 0)
      return fir::fnacc::FNACCBackendSupport::failure(
          "parallel subgroup and pipeline-stage counts must be positive");

    if (schedule.subgroupWidth != 32)
      return fir::fnacc::FNACCBackendSupport::failure(
          "Triton CUDA lowering requires subgroup width 32");

    for (auto [index, parameter] : llvm::enumerate(plan.abi.parameters))
      if (parameter.slot != index)
        return fir::fnacc::FNACCBackendSupport::failure(
            "kernel ABI slots are not contiguous");

    if (kernel.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D &&
        kernel.elementType != fir::fnacc::ElementType::F32 &&
        kernel.elementType != fir::fnacc::ElementType::F64)
      return fir::fnacc::FNACCBackendSupport::failure(
          "Triton matmul supports f32 and f64 element types");

    return fir::fnacc::FNACCBackendSupport::success();
  }

  void beginModule(const fir::fnacc::FNACCKernelPlanOptions &options,
                   llvm::raw_ostream &os) const override {
    os << "module attributes {"
       << "\"ttg.num-warps\" = " << options.requestedParallelSubgroups
       << " : i32, " << "\"ttg.num-ctas\" = 1 : i32, "
       << "\"ttg.num-stages\" = " << options.pipelineStages << " : i32, "
       << "\"ttg.threads-per-warp\" = " << options.subgroupWidth << " : i32"
       << "} {\n";
  }

  LogicalResult emitKernel(const fir::fnacc::FNACCKernelPlan &plan,
                           llvm::raw_ostream &os) const override {
    using Kind = fir::fnacc::ElementwiseKernelKind;
    const fir::fnacc::ElementwiseKernel &kernel = plan.kernel;
    int64_t blockX = plan.schedule.tile.x;
    int64_t blockY = plan.schedule.tile.y;
    int64_t blockZ = plan.schedule.tile.z;

    if (kernel.rank == 2) {
      if (kernel.kind == Kind::MatMul2D) {
        if (kernel.elementType == fir::fnacc::ElementType::F64) {
          switch (plan.schedule.f64MatmulStrategy) {
          case fir::fnacc::FNACCMatmulStrategy::Dot:
            emitTritonMatMul2DF64Dot(kernel, blockX, blockY, blockZ, plan.name,
                                     os);
            break;
          case fir::fnacc::FNACCMatmulStrategy::Reduce:
            emitTritonMatMul2DF64Reduce(kernel, blockX, blockY, blockZ,
                                        plan.name, os);
            break;
          case fir::fnacc::FNACCMatmulStrategy::FMA:
            emitTritonMatMul2DF64FMA(kernel, blockX, blockY, blockZ, plan.name,
                                     os);
            break;
          }
        } else {
          emitTritonMatMul2DF32(kernel, blockX, blockY, blockZ, plan.name, os);
        }
      } else if (kernel.kind == Kind::Expr2D) {
        emitTritonExpr2D(kernel, blockX, blockY, plan.name, os);
      } else {
        emitTriton2D(kernel, blockX, blockY, plan.name, os);
      }
    } else if (kernel.kind == Kind::ReductionDot1D) {
      emitTritonReductionDot1D(kernel, blockX, plan.name, os);
    } else if (fir::fnacc::isReductionKernelKind(kernel.kind)) {
      emitTritonReduction1D(kernel, blockX, plan.name, os);
    } else if (kernel.kind == Kind::Expr1D) {
      emitTritonExpr1D(kernel, blockX, plan.name, os);
    } else if (kernel.kind == Kind::Saxpy1D) {
      emitTritonSaxpy1D(kernel, blockX, plan.name, os);
    } else {
      emitTriton1D(kernel, blockX, plan.name, os);
    }

    if (plan.reductionStage)
      emitTritonReductionStage1D(plan.reductionStage->elementType,
                                 plan.reductionStage->reductionOperator, blockX,
                                 plan.reductionStage->name, os);

    return success();
  }

  void endModule(llvm::raw_ostream &os) const override { os << "}\n"; }

  int32_t getPrivatePointerArgumentCount(
      const fir::fnacc::FNACCKernelPlan &) const override {
    return 2;
  }
};

struct FNACCLowerToTritonPass
    : public fir::fnacc::impl::FNACCLowerToTritonBase<FNACCLowerToTritonPass> {
  FNACCLowerToTritonPass() = default;

  FNACCLowerToTritonPass(llvm::StringRef ttirOutput, llvm::StringRef jsonOutput,
                         int32_t numWarps, int32_t threadsPerWarp,
                         int32_t numStages, llvm::StringRef f64MatmulStrategy,
                         llvm::StringRef backend,
                         llvm::StringRef fallbackBackend,
                         bool allowBackendFallback) {
    this->ttirOutput = ttirOutput.str();
    this->jsonOutput = jsonOutput.str();
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numStages = numStages;
    this->f64MatmulStrategy = f64MatmulStrategy;
    this->backend = backend.str();
    this->fallbackBackend = fallbackBackend.str();
    this->allowBackendFallback = allowBackendFallback;
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

    if (!isValidBackendName(this->backend.getValue()) ||
        !isValidBackendName(this->fallbackBackend.getValue())) {
      module.emitError(
          "FNACC backend names may contain only letters, digits, '.', '_', "
          "'+' and '-'");
      signalPassFailure();
      return;
    }

    fir::fnacc::FNACCKernelPlanOptions planOptions;
    planOptions.requestedParallelSubgroups = tritonNumWarps;
    planOptions.subgroupWidth = tritonThreadsPerWarp;
    planOptions.pipelineStages = tritonNumStages;

    if (this->f64MatmulStrategy == "dot") {
      planOptions.f64MatmulStrategy = fir::fnacc::FNACCMatmulStrategy::Dot;
    } else if (this->f64MatmulStrategy == "reduce") {
      planOptions.f64MatmulStrategy = fir::fnacc::FNACCMatmulStrategy::Reduce;
    } else if (this->f64MatmulStrategy == "fma") {
      planOptions.f64MatmulStrategy = fir::fnacc::FNACCMatmulStrategy::FMA;
    } else {
      module.emitError("FNACC f64-matmul-strategy must be dot, reduce, or fma");
      signalPassFailure();
      return;
    }

    TritonBackend tritonBackend;
    llvm::SmallVector<const fir::fnacc::FNACCCodegenBackend *> backends{
        &tritonBackend};
    std::vector<fir::fnacc::FNACCKernelPlan> plans;
    std::vector<const fir::fnacc::FNACCCodegenBackend *> selectedBackends;
    bool usedBackendFallback = false;

    bool hasSimpleElementwiseLaunch = false;
    bool hasReductionLaunch = false;
    bool hasMatmulLaunch = false;
    bool planningFailed = false;

    int32_t nextSyntheticKernelId = 0;
    int32_t scanFallbackId = 0;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      int32_t kernelId = scanFallbackId++;
      if (auto attr = launchOp->getAttrOfType<IntegerAttr>("fnacc.kernel_id"))
        kernelId = static_cast<int32_t>(attr.getInt());
      nextSyntheticKernelId = std::max(nextSyntheticKernelId, kernelId + 1);
    });

    int32_t fallbackId = 0;

    module.walk([&](fir::fnacc::LaunchOp launchOp) {
      auto result = fir::fnacc::buildFNACCKernelPlan(
          launchOp, fallbackId++, nextSyntheticKernelId, planOptions);
      if (result.failed()) {
        launchOp.emitError("FNACC cannot plan launch: ")
            << result.getFailure().reason;
        planningFailed = true;
        return;
      }

      fir::fnacc::FNACCKernelPlan plan = result.takePlan();
      const fir::fnacc::ElementwiseKernel &k = plan.kernel;

      if (plan.reductionStage)
        ++nextSyntheticKernelId;

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

      fir::fnacc::FNACCBackendSelection selection =
          fir::fnacc::selectFNACCBackend(plan, backends, this->backend,
                                         this->fallbackBackend,
                                         this->allowBackendFallback);
      if (!selection.succeeded()) {
        launchOp.emitError("FNACC backend selection failed: ")
            << selection.diagnostic;
        planningFailed = true;
        return;
      }

      if (selection.usedFallback)
        launchOp.emitWarning() << selection.diagnostic;
      usedBackendFallback |= selection.usedFallback;

      selectedBackends.push_back(selection.backend);
      plans.push_back(std::move(plan));
    });

    if (planningFailed) {
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

    const fir::fnacc::FNACCCodegenBackend *moduleBackend =
        selectedBackends.empty() ? &tritonBackend : selectedBackends.front();

    jsonOs << "{\n";
    jsonOs << "  \"fnacc_schema_version\": 1,\n";
    jsonOs << "  \"backend_contract_version\": 1,\n";
    jsonOs << "  \"requested_backend\": \"" << this->backend.getValue()
           << "\",\n";
    jsonOs << "  \"fallback_backend\": \"" << this->fallbackBackend.getValue()
           << "\",\n";
    jsonOs << "  \"allow_backend_fallback\": "
           << (this->allowBackendFallback.getValue() ? "true" : "false")
           << ",\n";
    jsonOs << "  \"used_backend_fallback\": "
           << (usedBackendFallback ? "true" : "false") << ",\n";
    jsonOs << "  \"selected_backend\": \"" << moduleBackend->getName()
           << "\",\n";
    jsonOs << "  \"device_ir_kind\": \""
           << fir::fnacc::fnaccDeviceIRKindName(
                  moduleBackend->getDeviceIRKind())
           << "\",\n";
    jsonOs << "  \"device_image_kind\": \""
           << fir::fnacc::fnaccDeviceImageKindName(
                  moduleBackend->getRuntimeImageKind())
           << "\",\n";
    jsonOs << "  \"kernels\": [\n";

    moduleBackend->beginModule(planOptions, ttirOs);

    bool firstKernel = true;
    bool emissionFailed = false;

    int32_t emittedPtxIndex = 0;

    for (auto it : llvm::enumerate(plans)) {
      fir::fnacc::FNACCKernelPlan &plan = it.value();
      const fir::fnacc::FNACCCodegenBackend *backend =
          selectedBackends[it.index()];

      if (emissionFailed)
        break;

      if (backend != moduleBackend) {
        plan.launchOp.emitError(
            "FNACC mixed-backend module emission is not available yet");
        emissionFailed = true;
        break;
      }

      if (mlir::failed(backend->emitKernel(plan, ttirOs))) {
        plan.launchOp.emitError("FNACC backend '")
            << backend->getName() << "' failed while emitting kernel '"
            << plan.name << "'";
        emissionFailed = true;
        break;
      }

      emitJsonDescriptor(plan, emittedPtxIndex, *backend, jsonOs, firstKernel);
      ++emittedPtxIndex;

      if (plan.reductionStage) {
        emitJsonReductionStageDescriptor(plan, emittedPtxIndex, *backend,
                                         jsonOs, firstKernel);
        ++emittedPtxIndex;
      }
    }

    moduleBackend->endModule(ttirOs);

    jsonOs << "\n";
    jsonOs << "  ]\n";
    jsonOs << "}\n";

    if (emissionFailed) {
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
    llvm::StringRef f64MatmulStrategy, llvm::StringRef backend,
    llvm::StringRef fallbackBackend, bool allowBackendFallback) {
  return std::make_unique<FNACCLowerToTritonPass>(
      ttirOutput, jsonOutput, numWarps, threadsPerWarp, numStages,
      f64MatmulStrategy, backend, fallbackBackend, allowBackendFallback);
}

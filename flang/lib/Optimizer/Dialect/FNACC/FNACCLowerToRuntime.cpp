#include "flang/Optimizer/Dialect/FNACC/FNACCDialect.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCKernelAnalysis.h"
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace fir::fnacc {
#define GEN_PASS_DEF_FNACCLOWERTORUNTIME
#include "flang/Optimizer/Dialect/FNACC/FNACCPasses.h.inc"
} // namespace fir::fnacc

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kKernelIdAttrName = "fnacc.kernel_id";

struct BlockShape {
  int32_t rank = 1;
  int32_t x = 1024;
  int32_t y = 1;
  int32_t z = 1;
};

static Type getMLIRElementType(OpBuilder &builder,
                               fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::I8:
    return builder.getI8Type();
  case fir::fnacc::ElementType::I16:
    return builder.getI16Type();
  case fir::fnacc::ElementType::I32:
    return builder.getI32Type();
  case fir::fnacc::ElementType::I64:
    return builder.getI64Type();
  case fir::fnacc::ElementType::F32:
    return builder.getF32Type();
  case fir::fnacc::ElementType::F64:
    return builder.getF64Type();
  default:
    llvm_unreachable("unsupported FNACC element type");
  }
}

static StringRef getElementwiseRuntimeName(fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::I8:
    return "__fnacc_launch_i8_v1";
  case fir::fnacc::ElementType::I16:
    return "__fnacc_launch_i16_v1";
  case fir::fnacc::ElementType::I32:
    return "__fnacc_launch_i32_v1";
  case fir::fnacc::ElementType::I64:
    return "__fnacc_launch_i64_v1";
  case fir::fnacc::ElementType::F32:
    return "__fnacc_launch_f32_v1";
  case fir::fnacc::ElementType::F64:
    return "__fnacc_launch_f64_v1";
  case fir::fnacc::ElementType::Unknown:
    llvm_unreachable("unknown FNACC element type");
  }
  llvm_unreachable("unhandled FNACC element type");
}

static StringRef getReductionRuntimeName(fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::I8:
    return "__fnacc_launch_reduce_i8_v2";
  case fir::fnacc::ElementType::I16:
    return "__fnacc_launch_reduce_i16_v2";
  case fir::fnacc::ElementType::I32:
    return "__fnacc_launch_reduce_i32_v2";
  case fir::fnacc::ElementType::I64:
    return "__fnacc_launch_reduce_i64_v2";
  case fir::fnacc::ElementType::F32:
    return "__fnacc_launch_reduce_f32_v2";
  case fir::fnacc::ElementType::F64:
    return "__fnacc_launch_reduce_f64_v2";
  case fir::fnacc::ElementType::Unknown:
    llvm_unreachable("unknown FNACC reduction element type");
  }
  llvm_unreachable("unhandled FNACC reduction element type");
}

static Value createZeroElement(OpBuilder &builder, Location loc, Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type))
    return arith::ConstantOp::create(builder, loc, type,
                                     builder.getIntegerAttr(integerType, 0));

  auto floatType = cast<FloatType>(type);
  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getFloatAttr(floatType, 0.0));
}

static BlockShape getBlockShape(fir::fnacc::LaunchOp launchOp,
                                const fir::fnacc::ElementwiseKernel &k) {
  BlockShape shape;

  llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

  if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
    shape.rank = 2;
    shape.x = tiles.size() >= 1 ? static_cast<int32_t>(tiles[0]) : 16;
    shape.y = tiles.size() >= 2 ? static_cast<int32_t>(tiles[1]) : 16;

    shape.z = tiles.size() >= 3
                  ? static_cast<int32_t>(tiles[2])
                  : (k.elementType == fir::fnacc::ElementType::F64 ? 8 : 32);

    return shape;
  }

  if (k.rank == 2) {
    shape.rank = 2;
    shape.x = tiles.size() >= 1 ? static_cast<int32_t>(tiles[0]) : 16;
    shape.y = tiles.size() >= 2 ? static_cast<int32_t>(tiles[1]) : 16;
    shape.z = 1;
    return shape;
  }

  shape.rank = 1;
  shape.x = tiles.size() >= 1 ? static_cast<int32_t>(tiles[0]) : 1024;
  shape.y = 1;
  shape.z = 1;
  return shape;
}

static int32_t getKernelId(fir::fnacc::LaunchOp launchOp, int32_t fallbackId) {
  if (auto attr = launchOp->getAttrOfType<IntegerAttr>(kKernelIdAttrName))
    return static_cast<int32_t>(attr.getInt());

  launchOp.emitWarning("fnacc.launch has no fnacc.kernel_id attribute; using "
                       "walk-order fallback");

  return fallbackId;
}

static func::FuncOp getOrCreateRuntimeDecl(ModuleOp module, OpBuilder &builder,
                                           Location loc, StringRef name,
                                           TypeRange argTypes) {
  auto fnType = builder.getFunctionType(argTypes, TypeRange{});

  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    if (existing.getFunctionType() != fnType) {
      existing.emitError("existing FNACC runtime declaration has incompatible "
                         "function type for symbol ")
          << name << ". Existing type is " << existing.getFunctionType()
          << ", requested type is " << fnType;
    }

    return existing;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto fn = func::FuncOp::create(loc, name, fnType);
  fn.setPrivate();

  builder.insert(fn);
  return fn;
}

static Type getI8RefType(OpBuilder &builder) {
  return fir::ReferenceType::get(builder.getI8Type());
}

static std::optional<fir::BoxType> getBoxTypeFromBoxLike(Type type) {
  if (auto boxTy = dyn_cast<fir::BoxType>(type))
    return boxTy;

  if (auto refTy = dyn_cast<fir::ReferenceType>(type)) {
    if (auto boxTy = dyn_cast<fir::BoxType>(refTy.getEleTy()))
      return boxTy;
  }

  return std::nullopt;
}

static Value loadBoxIfNeeded(OpBuilder &builder, Location loc, Value value) {
  auto refTy = dyn_cast<fir::ReferenceType>(value.getType());
  if (refTy && isa<fir::BoxType>(refTy.getEleTy()))
    return fir::LoadOp::create(builder, loc, value);

  return value;
}

static Value getAddressOfBoxIfNeeded(OpBuilder &builder, Location loc,
                                     Value value) {
  value = loadBoxIfNeeded(builder, loc, value);

  if (auto boxTy = dyn_cast<fir::BoxType>(value.getType())) {
    Type addrTy = boxTy.getEleTy();
    if (!isa<fir::HeapType, fir::PointerType>(addrTy))
      addrTy = fir::ReferenceType::get(addrTy);
    return fir::BoxAddrOp::create(builder, loc, addrTy, value);
  }

  return value;
}

static Value convertToOpaqueRuntimePtr(OpBuilder &builder, Location loc,
                                       Value value) {
  value = getAddressOfBoxIfNeeded(builder, loc, value);

  Type i8RefTy = getI8RefType(builder);

  if (value.getType() == i8RefTy)
    return value;

  return fir::ConvertOp::create(builder, loc, i8RefTy, value);
}

static Value convertToI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(64))
    return value;

  return fir::ConvertOp::create(builder, loc, builder.getI64Type(), value);
}

static Value constantI64(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 64);
}

static Value constantI32(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 32);
}

static Value convertToI32(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(32))
    return value;

  return fir::ConvertOp::create(builder, loc, builder.getI32Type(), value);
}

static bool isDefinedInsideLaunch(fir::fnacc::LaunchOp launchOp, Value value) {
  if (Operation *definingOp = value.getDefiningOp())
    return launchOp->isProperAncestor(definingOp);

  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument)
    return false;

  Operation *parentOp = blockArgument.getOwner()->getParentOp();
  return parentOp && (parentOp == launchOp.getOperation() ||
                      launchOp->isProperAncestor(parentOp));
}

static Value stripLaunchLocalFirConverts(fir::fnacc::LaunchOp launchOp,
                                         Value value) {
  while (auto convert = value.getDefiningOp<fir::ConvertOp>()) {
    if (!launchOp->isProperAncestor(convert.getOperation()))
      break;
    value = convert.getValue();
  }

  return value;
}

/// Return an array-like value that remains valid after `launchOp` is erased.
///
/// Kernel analysis deliberately retains the exact FIR array base used inside
/// fnacc.launch. For allocatables, that base is commonly a launch-local
/// fir.box_addr(fir.load(%descriptor)). Runtime lowering must rematerialize
/// that chain outside the launch instead of retaining an operand owned by the
/// region that it is about to erase.
static std::optional<Value>
getRuntimeVisibleArrayLike(fir::fnacc::LaunchOp launchOp, Value arrayLike) {
  if (!arrayLike)
    return std::nullopt;

  arrayLike = stripLaunchLocalFirConverts(launchOp, arrayLike);
  if (!isDefinedInsideLaunch(launchOp, arrayLike))
    return arrayLike;

  auto boxAddr = arrayLike.getDefiningOp<fir::BoxAddrOp>();
  if (!boxAddr)
    return std::nullopt;

  Value box = stripLaunchLocalFirConverts(launchOp, boxAddr->getOperand(0));
  if (!isDefinedInsideLaunch(launchOp, box))
    return box;

  auto load = box.getDefiningOp<fir::LoadOp>();
  if (!load)
    return std::nullopt;

  Value descriptorRef = stripLaunchLocalFirConverts(launchOp, load.getMemref());
  if (isDefinedInsideLaunch(launchOp, descriptorRef))
    return std::nullopt;

  auto refTy = dyn_cast<fir::ReferenceType>(descriptorRef.getType());
  if (!refTy || !isa<fir::BoxType>(refTy.getEleTy()))
    return std::nullopt;

  return descriptorRef;
}

static Value
materializeExtentValue(OpBuilder &builder, Location loc,
                       fir::fnacc::LaunchOp launchOp,
                       const fir::fnacc::ElementwiseExtentSource &source) {
  using Kind = fir::fnacc::ElementwiseExtentSourceKind;

  switch (source.kind) {
  case Kind::LoadIntegerRef: {
    if (isDefinedInsideLaunch(launchOp, source.value)) {
      launchOp.emitError(
          "FNACC loop extent reference is defined inside fnacc.launch");
      return {};
    }
    Value loaded = fir::LoadOp::create(builder, loc, source.value);
    return convertToI32(builder, loc, loaded);
  }

  case Kind::BoxDim: {
    if (isDefinedInsideLaunch(launchOp, source.value)) {
      launchOp.emitError(
          "FNACC loop extent descriptor is defined inside fnacc.launch");
      return {};
    }
    Value dim = arith::ConstantIndexOp::create(builder, loc, source.dim);

    // fir.box_dims returns lower bound, extent, stride.
    auto dims = fir::BoxDimsOp::create(builder, loc, source.value, dim);

    Value extent = dims.getExtent();
    return convertToI32(builder, loc, extent);
  }

  case Kind::Value:
    if (isDefinedInsideLaunch(launchOp, source.value)) {
      Value value = stripLaunchLocalFirConverts(launchOp, source.value);
      if (auto load = value.getDefiningOp<fir::LoadOp>()) {
        Value memref = stripLaunchLocalFirConverts(launchOp, load.getMemref());
        auto refTy = dyn_cast<fir::ReferenceType>(memref.getType());
        if (refTy) {
          Type elementType = refTy.getEleTy();
          bool supportedInteger =
              elementType.isInteger(8) || elementType.isInteger(16) ||
              elementType.isInteger(32) || elementType.isInteger(64);

          if (!isDefinedInsideLaunch(launchOp, memref) && supportedInteger) {
            Value loaded = fir::LoadOp::create(builder, loc, memref);
            return convertToI32(builder, loc, loaded);
          }
        }
      }

      launchOp.emitError(
          "FNACC loop extent value cannot be rematerialized outside "
          "fnacc.launch");
      return {};
    }
    return convertToI32(builder, loc, source.value);

  case Kind::Unknown:
    llvm_unreachable("unknown FNACC extent source");
  }

  llvm_unreachable("unhandled FNACC extent source");
}

static std::optional<int64_t> getElementByteSize(Type elementType) {
  if (elementType.isF32())
    return 4;

  if (elementType.isF64())
    return 8;

  if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    unsigned width = intTy.getWidth();
    if (width % 8 == 0)
      return width / 8;
  }

  return std::nullopt;
}

static Value getRuntimeElementPointer(OpBuilder &builder, Location loc,
                                      fir::fnacc::LaunchOp launchOp,
                                      Value arrayLike,
                                      fir::fnacc::ElementType type) {
  if (!arrayLike) {
    mlir::emitError(loc) << "FNACC internal error: null array-like value";
    return {};
  }
  std::optional<Value> runtimeVisible =
      getRuntimeVisibleArrayLike(launchOp, arrayLike);
  if (!runtimeVisible) {
    launchOp.emitError(
        "FNACC array address cannot be rematerialized outside fnacc.launch");
    return {};
  }

  Type elementTy = getMLIRElementType(builder, type);
  Type elementRefTy = fir::ReferenceType::get(elementTy);

  arrayLike = getAddressOfBoxIfNeeded(builder, loc, *runtimeVisible);

  return fir::ConvertOp::create(builder, loc, elementRefTy, arrayLike);
}

static Value getRuntimeScalarElementPointer(OpBuilder &builder, Location loc,
                                            Value scalarRef,
                                            fir::fnacc::ElementType type) {
  Type elementTy = getMLIRElementType(builder, type);
  Type elementRefTy = fir::ReferenceType::get(elementTy);

  if (scalarRef.getType() == elementRefTy)
    return scalarRef;

  return fir::ConvertOp::create(builder, loc, elementRefTy, scalarRef);
}

static void createRuntimeCall(ModuleOp module, OpBuilder &builder, Location loc,
                              StringRef runtimeName, ValueRange operands) {
  llvm::SmallVector<Type> argTypes;
  for (Value operand : operands)
    argTypes.push_back(operand.getType());

  func::FuncOp callee =
      getOrCreateRuntimeDecl(module, builder, loc, runtimeName, argTypes);

  func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                       operands);
}

struct FNACCContiguousArrayDescriptorArgs {
  // ABI:
  //   ptr, element_bytes, rank,
  //   extent0, extent1, extent2,
  //   byte_stride0, byte_stride1, byte_stride2
  llvm::SmallVector<Value, 9> values;
};

static void
createDescriptorRuntimeCall(ModuleOp module, OpBuilder &builder, Location loc,
                            StringRef runtimeName,
                            const FNACCContiguousArrayDescriptorArgs &desc) {
  createRuntimeCall(module, builder, loc, runtimeName, desc.values);
}

static std::optional<FNACCContiguousArrayDescriptorArgs>
tryCreateContiguousArrayDescriptorArgs(OpBuilder &builder, Location loc,
                                       Value arrayLike) {
  std::optional<fir::BoxType> boxTy =
      getBoxTypeFromBoxLike(arrayLike.getType());
  if (!boxTy)
    return std::nullopt;

  Type storageType = boxTy->getEleTy();
  while (true) {
    if (auto heapTy = dyn_cast<fir::HeapType>(storageType)) {
      storageType = heapTy.getEleTy();
      continue;
    }

    if (auto pointerTy = dyn_cast<fir::PointerType>(storageType)) {
      storageType = pointerTy.getEleTy();
      continue;
    }

    break;
  }

  auto seqTy = dyn_cast<fir::SequenceType>(storageType);
  if (!seqTy)
    return std::nullopt;

  unsigned rank = seqTy.getDimension();

  // This ABI v1 supports rank 1-3.
  if (rank < 1 || rank > 3)
    return std::nullopt;

  std::optional<int64_t> elementBytes = getElementByteSize(seqTy.getEleTy());
  if (!elementBytes)
    return std::nullopt;

  FNACCContiguousArrayDescriptorArgs desc;

  Value boxValue = loadBoxIfNeeded(builder, loc, arrayLike);
  Value ptr = convertToOpaqueRuntimePtr(builder, loc, boxValue);
  Value elementBytesValue = constantI64(builder, loc, *elementBytes);
  Value rankValue = constantI32(builder, loc, static_cast<int32_t>(rank));

  desc.values.push_back(ptr);
  desc.values.push_back(elementBytesValue);
  desc.values.push_back(rankValue);

  llvm::SmallVector<Value, 3> extents;
  llvm::SmallVector<Value, 3> strides;

  for (unsigned dim = 0; dim < rank; ++dim) {
    Value dimValue = arith::ConstantIndexOp::create(builder, loc, dim);

    auto dims = fir::BoxDimsOp::create(builder, loc, boxValue, dimValue);

    // fir.box_dims returns:
    //   result #0 = lower bound
    //   result #1 = extent
    //   result #2 = byte stride
    Value extent = dims->getResult(1);
    Value byteStride = dims->getResult(2);

    extents.push_back(convertToI64(builder, loc, extent));
    strides.push_back(convertToI64(builder, loc, byteStride));
  }

  // Pad extents to rank 3.
  for (unsigned dim = rank; dim < 3; ++dim)
    extents.push_back(constantI64(builder, loc, 1));

  // Pad strides to rank 3. Padded dimensions are ignored by the runtime.
  for (unsigned dim = rank; dim < 3; ++dim)
    strides.push_back(constantI64(builder, loc, 0));

  for (Value extent : extents)
    desc.values.push_back(extent);

  for (Value stride : strides)
    desc.values.push_back(stride);

  return desc;
}

static void validateContiguousBoxForLaunch(ModuleOp module, OpBuilder &builder,
                                           Location loc, Value arrayLike) {
  if (!arrayLike)
    return;

  if (auto desc =
          tryCreateContiguousArrayDescriptorArgs(builder, loc, arrayLike)) {
    createDescriptorRuntimeCall(module, builder, loc,
                                "__fnacc_validate_contiguous_desc", *desc);
  }
}

struct FNACCByteSizedArgs {
  // ABI:
  //   ptr, bytes
  llvm::SmallVector<Value, 2> values;
};

static Value stripFirConvert(Value value) {
  while (auto convert = value.getDefiningOp<fir::ConvertOp>())
    value = convert.getValue();
  return value;
}

static std::optional<Type> getReferencedObjectType(Value value) {
  Type type = value.getType();

  if (auto refTy = dyn_cast<fir::ReferenceType>(type))
    return refTy.getEleTy();

  if (auto heapTy = dyn_cast<fir::HeapType>(type))
    return heapTy.getEleTy();

  if (auto ptrTy = dyn_cast<fir::PointerType>(type))
    return ptrTy.getEleTy();

  return std::nullopt;
}

static bool isSupportedScalarByteSizedType(Type type) {
  return type.isF32() || type.isF64() || type.isInteger(8) ||
         type.isInteger(16) || type.isInteger(32) || type.isInteger(64);
}

static std::optional<int64_t> getScalarOrElementByteSize(Type type) {
  if (type.isF32())
    return 4;

  if (type.isF64())
    return 8;

  if (auto intTy = dyn_cast<IntegerType>(type)) {
    unsigned width = intTy.getWidth();
    if (width % 8 == 0)
      return width / 8;
  }

  return std::nullopt;
}

static std::optional<llvm::SmallVector<Value, 3>>
tryGetExtentsFromShapeValue(Value shapeValue) {
  if (!shapeValue)
    return std::nullopt;

  shapeValue = stripFirConvert(shapeValue);

  if (auto shapeOp = shapeValue.getDefiningOp<fir::ShapeOp>()) {
    llvm::SmallVector<Value, 3> extents;
    for (Value extent : shapeOp.getExtents())
      extents.push_back(extent);
    return extents;
  }

  if (auto shapeShiftOp = shapeValue.getDefiningOp<fir::ShapeShiftOp>()) {
    llvm::SmallVector<Value, 3> extents;
    for (Value extent : shapeShiftOp.getExtents())
      extents.push_back(extent);
    return extents;
  }

  return std::nullopt;
}

static std::optional<llvm::SmallVector<Value, 3>>
tryGetDeclareShapeExtents(Value value) {
  value = stripFirConvert(value);

  if (auto declareOp = value.getDefiningOp<fir::DeclareOp>()) {
    Value shape = declareOp.getShape();
    if (!shape)
      return std::nullopt;

    return tryGetExtentsFromShapeValue(shape);
  }

  return std::nullopt;
}

static std::optional<llvm::SmallVector<Value, 3>>
tryCreateStaticSequenceExtents(OpBuilder &builder, Location loc,
                               fir::SequenceType seqTy) {
  llvm::SmallVector<Value, 3> extents;

  for (int64_t extent : seqTy.getShape()) {
    if (extent == fir::SequenceType::getUnknownExtent())
      return std::nullopt;

    extents.push_back(constantI64(builder, loc, extent));
  }

  return extents;
}

static std::optional<FNACCByteSizedArgs>
tryCreateByteSizedRuntimeArgs(OpBuilder &builder, Location loc, Value var) {
  Value value = stripFirConvert(var);
  Value baseAddress = value;

  // If this is a fir.declare result, use the declared memref as the base
  // pointer, and use its shape operand to recover explicit-shape extents.
  llvm::SmallVector<Value, 3> declareExtents;
  bool hasDeclareExtents = false;

  if (auto declareOp = value.getDefiningOp<fir::DeclareOp>()) {
    baseAddress = declareOp.getMemref();

    if (auto extents = tryGetDeclareShapeExtents(value)) {
      declareExtents = *extents;
      hasDeclareExtents = true;
    }
  }

  std::optional<Type> objectType = getReferencedObjectType(baseAddress);
  if (!objectType)
    return std::nullopt;

  FNACCByteSizedArgs args;

  // Scalar case.
  if (!isa<fir::SequenceType>(*objectType)) {
    if (!isSupportedScalarByteSizedType(*objectType))
      return std::nullopt;

    std::optional<int64_t> bytes = getScalarOrElementByteSize(*objectType);
    if (!bytes)
      return std::nullopt;

    Value ptr = convertToOpaqueRuntimePtr(builder, loc, baseAddress);
    args.values.push_back(ptr);
    args.values.push_back(constantI64(builder, loc, *bytes));
    return args;
  }

  // Array case.
  auto seqTy = dyn_cast<fir::SequenceType>(*objectType);
  if (!seqTy)
    return std::nullopt;

  unsigned rank = seqTy.getDimension();
  if (rank < 1 || rank > 3)
    return std::nullopt;

  std::optional<int64_t> elementBytes =
      getScalarOrElementByteSize(seqTy.getEleTy());
  if (!elementBytes)
    return std::nullopt;

  llvm::SmallVector<Value, 3> extents;

  if (hasDeclareExtents) {
    extents = declareExtents;
  } else if (auto staticExtents =
                 tryCreateStaticSequenceExtents(builder, loc, seqTy)) {
    extents = *staticExtents;
  } else {
    // Dynamic explicit-shape arrays usually require fir.declare shape
    // information. If it was not available, this helper cannot safely size
    // the object.
    return std::nullopt;
  }

  if (extents.size() != rank)
    return std::nullopt;

  Value ptr = convertToOpaqueRuntimePtr(builder, loc, baseAddress);
  Value bytesValue = constantI64(builder, loc, *elementBytes);

  for (Value extent : extents) {
    Value extentI64 = convertToI64(builder, loc, extent);
    bytesValue = arith::MulIOp::create(builder, loc, bytesValue, extentI64);
  }

  args.values.push_back(ptr);
  args.values.push_back(bytesValue);
  return args;
}

static LogicalResult lowerFNACCDataOpsToRuntime(ModuleOp module,
                                                OpBuilder &builder) {
  llvm::SmallVector<Operation *> dataOps;

  module.walk([&](Operation *op) {
    if (isa<fir::fnacc::UpdateHostOp, fir::fnacc::UpdateDeviceOp,
            fir::fnacc::ReleaseOp, fir::fnacc::ReleaseAllOp,
            fir::fnacc::CopyinOp, fir::fnacc::CreateOp, fir::fnacc::CopyoutOp,
            fir::fnacc::DeleteOp, fir::fnacc::WaitOp>(op)) {
      dataOps.push_back(op);
    }
  });

  auto lowerCopyToDevice = [&](Operation *op, Value var) -> LogicalResult {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);

    if (auto desc = tryCreateContiguousArrayDescriptorArgs(builder, loc, var)) {
      createDescriptorRuntimeCall(module, builder, loc,
                                  "__fnacc_update_device_desc", *desc);
    } else if (auto bytes = tryCreateByteSizedRuntimeArgs(builder, loc, var)) {
      createRuntimeCall(module, builder, loc, "__fnacc_update_device_bytes",
                        bytes->values);
    } else {
      op->emitWarning()
          << "FNACC update/copyin device could not determine object size; "
             "falling back to existing-allocation raw pointer update";
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);
      createRuntimeCall(module, builder, loc, "__fnacc_update_device",
                        ValueRange{ptr});
    }

    op->erase();
    return success();
  };

  auto lowerCreate = [&](Operation *op, Value var) -> LogicalResult {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);

    if (auto desc = tryCreateContiguousArrayDescriptorArgs(builder, loc, var)) {
      createDescriptorRuntimeCall(module, builder, loc, "__fnacc_create_desc",
                                  *desc);
    } else if (auto bytes = tryCreateByteSizedRuntimeArgs(builder, loc, var)) {
      createRuntimeCall(module, builder, loc, "__fnacc_create_bytes",
                        bytes->values);
    } else {
      op->emitError()
          << "FNACC create could not determine object size; create requires "
             "a sized scalar, explicit-shape array, or descriptor";
      return failure();
    }

    op->erase();
    return success();
  };

  auto lowerCopyToHost = [&](Operation *op, Value var) -> LogicalResult {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);

    if (auto desc = tryCreateContiguousArrayDescriptorArgs(builder, loc, var)) {
      createDescriptorRuntimeCall(module, builder, loc,
                                  "__fnacc_update_host_desc", *desc);
    } else if (auto bytes = tryCreateByteSizedRuntimeArgs(builder, loc, var)) {
      createRuntimeCall(module, builder, loc, "__fnacc_update_host_bytes",
                        bytes->values);
    } else {
      op->emitWarning()
          << "FNACC update/copyout host could not determine object size; "
             "falling back to existing-allocation raw pointer update";
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);
      createRuntimeCall(module, builder, loc, "__fnacc_update_host",
                        ValueRange{ptr});
    }

    op->erase();
    return success();
  };

  auto lowerReleaseOne = [&](Operation *op, Value var) -> LogicalResult {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);

    Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);

    if (getBoxTypeFromBoxLike(var.getType())) {
      createRuntimeCall(module, builder, loc, "__fnacc_release_desc",
                        ValueRange{ptr});
    } else {
      createRuntimeCall(module, builder, loc, "__fnacc_release",
                        ValueRange{ptr});
    }

    return success();
  };

  for (Operation *op : dataOps) {
    if (auto updateHost = dyn_cast<fir::fnacc::UpdateHostOp>(op)) {
      if (failed(lowerCopyToHost(op, updateHost.getVar())))
        return failure();
      continue;
    }

    if (auto updateDevice = dyn_cast<fir::fnacc::UpdateDeviceOp>(op)) {
      if (failed(lowerCopyToDevice(op, updateDevice.getVar())))
        return failure();
      continue;
    }

    if (auto copyin = dyn_cast<fir::fnacc::CopyinOp>(op)) {
      if (failed(lowerCopyToDevice(op, copyin.getVar())))
        return failure();
      continue;
    }

    if (auto create = dyn_cast<fir::fnacc::CreateOp>(op)) {
      if (failed(lowerCreate(op, create.getVar())))
        return failure();
      continue;
    }

    if (auto copyout = dyn_cast<fir::fnacc::CopyoutOp>(op)) {
      if (failed(lowerCopyToHost(op, copyout.getVar())))
        return failure();
      continue;
    }

    if (auto del = dyn_cast<fir::fnacc::DeleteOp>(op)) {
      if (failed(lowerReleaseOne(op, del.getVar())))
        return failure();
      op->erase();
      continue;
    }

    if (auto release = dyn_cast<fir::fnacc::ReleaseOp>(op)) {
      for (Value var : release.getVars()) {
        if (failed(lowerReleaseOne(op, var)))
          return failure();
      }

      op->erase();
      continue;
    }

    if (auto releaseAll = dyn_cast<fir::fnacc::ReleaseAllOp>(op)) {
      Location loc = releaseAll.getLoc();
      builder.setInsertionPoint(releaseAll);

      createRuntimeCall(module, builder, loc, "__fnacc_release_all",
                        ValueRange{});

      op->erase();
      continue;
    }

    if (auto wait = dyn_cast<fir::fnacc::WaitOp>(op)) {
      Location loc = wait.getLoc();
      builder.setInsertionPoint(wait);

      createRuntimeCall(module, builder, loc, "__fnacc_wait", ValueRange{});

      op->erase();
      continue;
    }
  }

  return success();
}

struct FNACCLowerToRuntimePass
    : public fir::fnacc::impl::FNACCLowerToRuntimeBase<
          FNACCLowerToRuntimePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    llvm::SmallVector<fir::fnacc::LaunchOp> launches;
    module.walk(
        [&](fir::fnacc::LaunchOp launchOp) { launches.push_back(launchOp); });

    bool recognitionFailed = false;
    for (fir::fnacc::LaunchOp launchOp : launches) {
      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);
      if (result.failed()) {
        launchOp.emitError("FNACC runtime cannot lower launch: ")
            << result.getFailure().reason;
        recognitionFailed = true;
      }
    }
    if (recognitionFailed) {
      signalPassFailure();
      return;
    }

    int32_t fallbackKernelId = 0;

    for (fir::fnacc::LaunchOp launchOp : launches) {
      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);

      if (result.failed()) {
        launchOp.emitError("FNACC runtime cannot lower launch: ")
            << result.getFailure().reason;
        signalPassFailure();
        return;
      }

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();

      Location loc = launchOp.getLoc();
      BlockShape blockShape = getBlockShape(launchOp, k);

      builder.setInsertionPoint(launchOp);

      // Kernel emitters currently linearize arrays and therefore require
      // contiguous storage. Preserve box strides long enough to reject a
      // non-contiguous assumed-shape actual argument before its base pointer
      // is passed to the CUDA runtime.
      for (Value readArray : k.readArrays)
        validateContiguousBoxForLaunch(module, builder, loc, readArray);
      validateContiguousBoxForLaunch(module, builder, loc, k.writeArray);

      int32_t stableKernelId = getKernelId(launchOp, fallbackKernelId);

      Value kernelIdValue =
          arith::ConstantIntOp::create(builder, loc, stableKernelId, 32);

      Value rankValue = arith::ConstantIntOp::create(builder, loc, k.rank, 32);

      Value blockXValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.x, 32);

      Value blockYValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.y, 32);

      Value blockZValue =
          arith::ConstantIntOp::create(builder, loc, blockShape.z, 32);

      Value extentXValue =
          materializeExtentValue(builder, loc, launchOp, k.extentX);

      Value extentYValue;
      if (k.rank == 2) {
        extentYValue =
            materializeExtentValue(builder, loc, launchOp, k.extentY);
      } else {
        extentYValue = arith::ConstantIntOp::create(builder, loc, 1, 32);
      }

      Value extentZValue;
      if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
        extentZValue =
            materializeExtentValue(builder, loc, launchOp, k.extentZ);
      } else {
        extentZValue = arith::ConstantIntOp::create(builder, loc, 1, 32);
      }

      if (!extentXValue || !extentYValue || !extentZValue) {
        signalPassFailure();
        return;
      }

      if (k.kind == fir::fnacc::ElementwiseKernelKind::ReductionSum1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionDot1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionProduct1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMin1D ||
          k.kind == fir::fnacc::ElementwiseKernelKind::ReductionMax1D) {
        if (!k.reductionScalarRef) {
          launchOp.emitError("FNACC reduction has no reduction scalar ref");
          signalPassFailure();
          return;
        }

        if (k.readArrays.empty() || k.readArrays.size() > 2) {
          launchOp.emitError(
              "FNACC reduction runtime supports one or two read arrays");
          signalPassFailure();
          return;
        }

        Value initialValue =
            fir::LoadOp::create(builder, loc, k.reductionScalarRef);

        StringRef reductionRuntimeName = getReductionRuntimeName(k.elementType);

        Value numReadArraysValue = arith::ConstantIntOp::create(
            builder, loc, static_cast<int32_t>(k.readArrays.size()), 32);

        Value read0Ptr = getRuntimeElementPointer(
            builder, loc, launchOp, k.readArrays[0], k.elementType);

        Value read1Ptr = read0Ptr;
        if (k.readArrays.size() >= 2) {
          read1Ptr = getRuntimeElementPointer(builder, loc, launchOp,
                                              k.readArrays[1], k.elementType);
        }

        if (!read0Ptr || !read1Ptr) {
          signalPassFailure();
          return;
        }

        Value resultPtr = getRuntimeScalarElementPointer(
            builder, loc, k.reductionScalarRef, k.elementType);

        llvm::SmallVector<Type> argTypes;
        argTypes.push_back(kernelIdValue.getType());
        argTypes.push_back(blockXValue.getType());
        argTypes.push_back(numReadArraysValue.getType());
        argTypes.push_back(read0Ptr.getType());
        argTypes.push_back(read1Ptr.getType());
        argTypes.push_back(resultPtr.getType());
        argTypes.push_back(initialValue.getType());
        argTypes.push_back(extentXValue.getType());

        func::FuncOp callee = getOrCreateRuntimeDecl(
            module, builder, loc, reductionRuntimeName, argTypes);

        llvm::SmallVector<Value> operands;
        operands.push_back(kernelIdValue);
        operands.push_back(blockXValue);
        operands.push_back(numReadArraysValue);
        operands.push_back(read0Ptr);
        operands.push_back(read1Ptr);
        operands.push_back(resultPtr);
        operands.push_back(initialValue);
        operands.push_back(extentXValue);

        func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                             operands);

        launchOp.erase();

        ++fallbackKernelId;
        continue;
      }

      StringRef runtimeName = getElementwiseRuntimeName(k.elementType);

      Type elementTy = getMLIRElementType(builder, k.elementType);
      Type elementRefTy = fir::ReferenceType::get(elementTy);

      if (k.readArrays.empty() || k.readArrays.size() > 3) {
        launchOp.emitError(
            "FNACC runtime lowering currently requires one to three "
            "read arrays");
        signalPassFailure();
        return;
      }

      Value read0Ptr = getRuntimeElementPointer(builder, loc, launchOp,
                                                k.readArrays[0], k.elementType);

      Value read1Ptr = read0Ptr;
      if (k.readArrays.size() >= 2)
        read1Ptr = getRuntimeElementPointer(builder, loc, launchOp,
                                            k.readArrays[1], k.elementType);

      Value read2Ptr = read0Ptr;
      if (k.readArrays.size() >= 3)
        read2Ptr = getRuntimeElementPointer(builder, loc, launchOp,
                                            k.readArrays[2], k.elementType);

      Value writePtr = getRuntimeElementPointer(builder, loc, launchOp,
                                                k.writeArray, k.elementType);

      if (!read0Ptr || !read1Ptr || !read2Ptr || !writePtr) {
        signalPassFailure();
        return;
      }

      if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {

        StringRef runtimeName = k.elementType == fir::fnacc::ElementType::F64
                                    ? "__fnacc_launch_matmul_f64_v1"
                                    : "__fnacc_launch_matmul_f32_v1";

        llvm::SmallVector<Type> argTypes;
        argTypes.push_back(kernelIdValue.getType());
        argTypes.push_back(blockXValue.getType());
        argTypes.push_back(blockYValue.getType());
        argTypes.push_back(blockZValue.getType());
        argTypes.push_back(read0Ptr.getType());
        argTypes.push_back(read1Ptr.getType());
        argTypes.push_back(writePtr.getType());
        argTypes.push_back(extentXValue.getType());
        argTypes.push_back(extentYValue.getType());
        argTypes.push_back(extentZValue.getType());

        func::FuncOp callee =
            getOrCreateRuntimeDecl(module, builder, loc, runtimeName, argTypes);

        llvm::SmallVector<Value> operands;
        operands.push_back(kernelIdValue);
        operands.push_back(blockXValue);
        operands.push_back(blockYValue);
        operands.push_back(blockZValue);
        operands.push_back(read0Ptr);
        operands.push_back(read1Ptr);
        operands.push_back(writePtr);
        operands.push_back(extentXValue);
        operands.push_back(extentYValue);
        operands.push_back(extentZValue);

        func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                             operands);

        launchOp.erase();

        ++fallbackKernelId;
        continue;
      }

      unsigned scalarCount = k.scalarRefs.size();

      if (k.readArrays.size() > 3) {
        launchOp.emitError(
            "FNACC generic runtime supports at most three read arrays");
        signalPassFailure();
        return;
      }

      if (scalarCount > 3) {
        launchOp.emitError(
            "FNACC generic runtime supports at most three scalar values");
        signalPassFailure();
        return;
      }

      Value numReadArraysValue =
          arith::ConstantIntOp::create(builder, loc, k.readArrays.size(), 32);

      Value numScalarsValue =
          arith::ConstantIntOp::create(builder, loc, scalarCount, 32);

      Value zeroElement = createZeroElement(builder, loc, elementTy);

      Value scalar0Value = zeroElement;
      Value scalar1Value = zeroElement;
      Value scalar2Value = zeroElement;

      if (scalarCount >= 1)
        scalar0Value = fir::LoadOp::create(builder, loc, k.scalarRefs[0]);

      if (scalarCount >= 2)
        scalar1Value = fir::LoadOp::create(builder, loc, k.scalarRefs[1]);

      if (scalarCount >= 3)
        scalar2Value = fir::LoadOp::create(builder, loc, k.scalarRefs[2]);

      llvm::SmallVector<Type> argTypes;
      argTypes.push_back(kernelIdValue.getType());
      argTypes.push_back(rankValue.getType());
      argTypes.push_back(blockXValue.getType());
      argTypes.push_back(blockYValue.getType());
      argTypes.push_back(blockZValue.getType());
      argTypes.push_back(numReadArraysValue.getType());
      argTypes.push_back(numScalarsValue.getType());
      argTypes.push_back(elementRefTy);
      argTypes.push_back(elementRefTy);
      argTypes.push_back(elementRefTy);
      argTypes.push_back(elementRefTy);
      argTypes.push_back(elementTy);
      argTypes.push_back(elementTy);
      argTypes.push_back(elementTy);
      argTypes.push_back(extentXValue.getType());
      argTypes.push_back(extentYValue.getType());
      argTypes.push_back(extentZValue.getType());

      func::FuncOp callee =
          getOrCreateRuntimeDecl(module, builder, loc, runtimeName, argTypes);

      llvm::SmallVector<Value> operands;
      operands.push_back(kernelIdValue);
      operands.push_back(rankValue);
      operands.push_back(blockXValue);
      operands.push_back(blockYValue);
      operands.push_back(blockZValue);
      operands.push_back(numReadArraysValue);
      operands.push_back(numScalarsValue);
      operands.push_back(read0Ptr);
      operands.push_back(read1Ptr);
      operands.push_back(read2Ptr);
      operands.push_back(writePtr);
      operands.push_back(scalar0Value);
      operands.push_back(scalar1Value);
      operands.push_back(scalar2Value);
      operands.push_back(extentXValue);
      operands.push_back(extentYValue);
      operands.push_back(extentZValue);

      func::CallOp::create(builder, loc, callee.getSymName(), TypeRange{},
                           operands);

      launchOp.erase();

      ++fallbackKernelId;
    }
    if (failed(lowerFNACCDataOpsToRuntime(module, builder))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCLowerToRuntimePass() {
  return std::make_unique<FNACCLowerToRuntimePass>();
}

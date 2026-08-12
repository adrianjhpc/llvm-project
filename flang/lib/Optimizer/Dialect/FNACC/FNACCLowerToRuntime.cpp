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
#include "llvm/ADT/SmallVector.h"

#include <optional>

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

static FloatType getMLIRFloatType(OpBuilder &builder,
                                  fir::fnacc::ElementType type) {
  switch (type) {
  case fir::fnacc::ElementType::F32:
    return builder.getF32Type();
  case fir::fnacc::ElementType::F64:
    return builder.getF64Type();
  default:
    llvm_unreachable("unsupported FNACC element type");
  }
}

static BlockShape getBlockShape(fir::fnacc::LaunchOp launchOp,
                                const fir::fnacc::ElementwiseKernel &k) {
  BlockShape shape;

  llvm::ArrayRef<int64_t> tiles = launchOp.getTileSizes();

  if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
    shape.rank = 2;
    shape.x = tiles.size() >= 1 ? static_cast<int32_t>(tiles[0]) : 16;
    shape.y = tiles.size() >= 2 ? static_cast<int32_t>(tiles[1]) : 16;
    shape.z = tiles.size() >= 3 ? static_cast<int32_t>(tiles[2]) : 32;
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

static Value getAddressOfBoxIfNeeded(OpBuilder &builder, Location loc,
                                     Value value) {
  if (auto boxTy = dyn_cast<fir::BoxType>(value.getType())) {
    Type addrTy = fir::ReferenceType::get(boxTy.getEleTy());
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

static Value
materializeExtentValue(OpBuilder &builder, Location loc,
                       const fir::fnacc::ElementwiseExtentSource &source) {
  using Kind = fir::fnacc::ElementwiseExtentSourceKind;

  switch (source.kind) {
  case Kind::LoadI32Ref: {
    Value loaded = fir::LoadOp::create(builder, loc, source.value);
    return convertToI32(builder, loc, loaded);
  }

  case Kind::BoxDim: {
    Value dim = arith::ConstantIndexOp::create(builder, loc, source.dim);

    // fir.box_dims returns lower bound, extent, stride.
    auto dims = fir::BoxDimsOp::create(builder, loc, source.value, dim);

    Value extent = dims.getExtent();
    return convertToI32(builder, loc, extent);
  }

  case Kind::Value:
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

static Value getRuntimeRealPointer(OpBuilder &builder, Location loc,
                                   Value arrayLike,
                                   fir::fnacc::ElementType type) {
  FloatType realTy = getMLIRFloatType(builder, type);
  Type realRefTy = fir::ReferenceType::get(realTy);

  if (auto boxTy = dyn_cast<fir::BoxType>(arrayLike.getType())) {
    Type addrTy = fir::ReferenceType::get(boxTy.getEleTy());
    Value baseAddr = fir::BoxAddrOp::create(builder, loc, addrTy, arrayLike);
    return fir::ConvertOp::create(builder, loc, realRefTy, baseAddr);
  }

  return fir::ConvertOp::create(builder, loc, realRefTy, arrayLike);
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
  auto boxTy = dyn_cast<fir::BoxType>(arrayLike.getType());
  if (!boxTy)
    return std::nullopt;

  auto seqTy = dyn_cast<fir::SequenceType>(boxTy.getEleTy());
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

  Value ptr = convertToOpaqueRuntimePtr(builder, loc, arrayLike);
  Value elementBytesValue = constantI64(builder, loc, *elementBytes);
  Value rankValue = constantI32(builder, loc, static_cast<int32_t>(rank));

  desc.values.push_back(ptr);
  desc.values.push_back(elementBytesValue);
  desc.values.push_back(rankValue);

  llvm::SmallVector<Value, 3> extents;
  llvm::SmallVector<Value, 3> strides;

  for (unsigned dim = 0; dim < rank; ++dim) {
    Value dimValue = arith::ConstantIndexOp::create(builder, loc, dim);

    auto dims = fir::BoxDimsOp::create(builder, loc, arrayLike, dimValue);

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

  Value ptr = convertToOpaqueRuntimePtr(builder, loc, baseAddress);

  // Scalar case.
  if (!isa<fir::SequenceType>(*objectType)) {
    if (!isSupportedScalarByteSizedType(*objectType))
      return std::nullopt;

    std::optional<int64_t> bytes = getScalarOrElementByteSize(*objectType);
    if (!bytes)
      return std::nullopt;

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

  Value bytesValue = constantI64(builder, loc, *elementBytes);

  for (Value extent : extents) {
    Value extentI64 = convertToI64(builder, loc, extent);
    bytesValue = arith::MulIOp::create(builder, loc, bytesValue, extentI64);
  }

  args.values.push_back(ptr);
  args.values.push_back(bytesValue);
  return args;
}

static void lowerFNACCDataOpsToRuntime(ModuleOp module, OpBuilder &builder) {
  llvm::SmallVector<fir::fnacc::UpdateHostOp> updateHostOps;
  llvm::SmallVector<fir::fnacc::UpdateDeviceOp> updateDeviceOps;
  llvm::SmallVector<fir::fnacc::ReleaseOp> releaseOps;
  llvm::SmallVector<fir::fnacc::ReleaseAllOp> releaseAllOps;

  module.walk(
      [&](fir::fnacc::UpdateHostOp op) { updateHostOps.push_back(op); });
  module.walk(
      [&](fir::fnacc::UpdateDeviceOp op) { updateDeviceOps.push_back(op); });
  module.walk([&](fir::fnacc::ReleaseOp op) { releaseOps.push_back(op); });
  module.walk(
      [&](fir::fnacc::ReleaseAllOp op) { releaseAllOps.push_back(op); });

  for (fir::fnacc::UpdateHostOp op : updateHostOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    Value var = op.getVar();

    if (auto desc = tryCreateContiguousArrayDescriptorArgs(builder, loc, var)) {
      createDescriptorRuntimeCall(module, builder, loc,
                                  "__fnacc_update_host_desc", *desc);
    } else if (auto bytes = tryCreateByteSizedRuntimeArgs(builder, loc, var)) {
      createRuntimeCall(module, builder, loc, "__fnacc_update_host_bytes",
                        bytes->values);
    } else {
      op.emitWarning()
          << "FNACC update host could not determine object size; falling "
             "back to existing-allocation raw pointer update";
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);
      createRuntimeCall(module, builder, loc, "__fnacc_update_host",
                        ValueRange{ptr});
    }

    op.erase();
  }

  for (fir::fnacc::UpdateDeviceOp op : updateDeviceOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    Value var = op.getVar();

    if (auto desc = tryCreateContiguousArrayDescriptorArgs(builder, loc, var)) {
      createDescriptorRuntimeCall(module, builder, loc,
                                  "__fnacc_update_device_desc", *desc);
    } else if (auto bytes = tryCreateByteSizedRuntimeArgs(builder, loc, var)) {
      createRuntimeCall(module, builder, loc, "__fnacc_update_device_bytes",
                        bytes->values);
    } else {
      op.emitWarning()
          << "FNACC update device could not determine object size; falling "
             "back to existing-allocation raw pointer update";
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);
      createRuntimeCall(module, builder, loc, "__fnacc_update_device",
                        ValueRange{ptr});
    }

    op.erase();
  }

  for (fir::fnacc::ReleaseOp op : releaseOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    for (Value var : op.getVars()) {
      Value ptr = convertToOpaqueRuntimePtr(builder, loc, var);

      if (isa<fir::BoxType>(var.getType())) {
        createRuntimeCall(module, builder, loc, "__fnacc_release_desc",
                          ValueRange{ptr});
      } else {
        createRuntimeCall(module, builder, loc, "__fnacc_release",
                          ValueRange{ptr});
      }
    }

    op.erase();
  }

  for (fir::fnacc::ReleaseAllOp op : releaseAllOps) {
    Location loc = op.getLoc();
    builder.setInsertionPoint(op);

    createRuntimeCall(module, builder, loc, "__fnacc_release_all",
                      ValueRange{});

    op.erase();
  }
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

    int32_t fallbackKernelId = 0;

    for (fir::fnacc::LaunchOp launchOp : launches) {
      auto result = fir::fnacc::recognizeElementwiseKernel(launchOp);

      if (result.failed()) {
        launchOp.emitWarning("FNACC runtime lowering skipped launch: ")
            << result.getFailure().reason;
        ++fallbackKernelId;
        continue;
      }

      const fir::fnacc::ElementwiseKernel &k = result.getKernel();

      Location loc = launchOp.getLoc();
      BlockShape blockShape = getBlockShape(launchOp, k);

      builder.setInsertionPoint(launchOp);

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

      Value extentXValue = materializeExtentValue(builder, loc, k.extentX);

      Value extentYValue;
      if (k.rank == 2) {
        extentYValue = materializeExtentValue(builder, loc, k.extentY);
      } else {
        extentYValue = arith::ConstantIntOp::create(builder, loc, 1, 32);
      }

      Value extentZValue;
      if (k.kind == fir::fnacc::ElementwiseKernelKind::MatMul2D) {
        extentZValue = materializeExtentValue(builder, loc, k.extentZ);
      } else {
        extentZValue = arith::ConstantIntOp::create(builder, loc, 1, 32);
      }

      StringRef runtimeName = k.elementType == fir::fnacc::ElementType::F64
                                  ? "__fnacc_launch_f64_v1"
                                  : "__fnacc_launch_f32_v1";

      FloatType realTy = getMLIRFloatType(builder, k.elementType);
      Type realRefTy = fir::ReferenceType::get(realTy);

      if (k.readArrays.empty() || k.readArrays.size() > 3) {
        launchOp.emitError(
            "FNACC runtime lowering currently requires one to three "
            "read arrays");
        signalPassFailure();
        return;
      }

      Value read0Ptr =
          getRuntimeRealPointer(builder, loc, k.readArrays[0], k.elementType);

      Value read1Ptr = read0Ptr;
      if (k.readArrays.size() >= 2)
        read1Ptr =
            getRuntimeRealPointer(builder, loc, k.readArrays[1], k.elementType);

      Value read2Ptr = read0Ptr;
      if (k.readArrays.size() >= 3)
        read2Ptr =
            getRuntimeRealPointer(builder, loc, k.readArrays[2], k.elementType);

      Value writePtr =
          getRuntimeRealPointer(builder, loc, k.writeArray, k.elementType);

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
            "FNACC generic runtime supports at most three f32 or f64 scalars");
        signalPassFailure();
        return;
      }

      Value numReadArraysValue =
          arith::ConstantIntOp::create(builder, loc, k.readArrays.size(), 32);

      Value numScalarsValue =
          arith::ConstantIntOp::create(builder, loc, scalarCount, 32);

      auto zeroAttr = builder.getFloatAttr(realTy, 0.0);

      Value zeroReal =
          arith::ConstantOp::create(builder, loc, realTy, zeroAttr);

      Value scalar0Value = zeroReal;
      Value scalar1Value = zeroReal;
      Value scalar2Value = zeroReal;

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
      argTypes.push_back(realRefTy);
      argTypes.push_back(realRefTy);
      argTypes.push_back(realRefTy);
      argTypes.push_back(realRefTy);
      argTypes.push_back(realTy);
      argTypes.push_back(realTy);
      argTypes.push_back(realTy);
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
    lowerFNACCDataOpsToRuntime(module, builder);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::fnacc::createFNACCLowerToRuntimePass() {
  return std::make_unique<FNACCLowerToRuntimePass>();
}

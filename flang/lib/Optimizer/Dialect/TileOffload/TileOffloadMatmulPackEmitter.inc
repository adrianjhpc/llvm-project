// Transpose the selected A(M,K) submatrix into AP(Kpad,M), Kpad a multiple
// of 16. Both loads and stores are tiled; addresses use 64-bit arithmetic.
// This helper is internal to LowerToTriton.cpp, not a separate translation
// unit.
static void emitMatmulPackA(StringRef name, llvm::raw_ostream &os) {
  os << "tt.func @" << name
     << "(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %m: i32, %k: i32, "
        "%ld: i32, %base: i64, %source_ld: i64) attributes {noinline = false} "
        "{\n";
  os << R"mlir(  %pm = tt.get_program_id x : i32
  %pk = tt.get_program_id y : i32
  %tile = arith.constant 32 : i32
  %bm = arith.muli %pm, %tile : i32
  %bk = arith.muli %pk, %tile : i32
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %bm_s = tt.splat %bm : i32 -> tensor<32xi32>
  %bk_s = tt.splat %bk : i32 -> tensor<32xi32>
  %im = arith.addi %bm_s, %r : tensor<32xi32>
  %ik = arith.addi %bk_s, %r : tensor<32xi32>
  %ms = tt.splat %m : i32 -> tensor<32xi32>
  %ks = tt.splat %k : i32 -> tensor<32xi32>
  %lds = tt.splat %ld : i32 -> tensor<32xi32>
  %mm = arith.cmpi slt, %im, %ms : tensor<32xi32>
  %mk = arith.cmpi slt, %ik, %ks : tensor<32xi32>
  %mp = arith.cmpi slt, %ik, %lds : tensor<32xi32>
  %mm_e = tt.expand_dims %mm {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
  %mk_e = tt.expand_dims %mk {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mp_e = tt.expand_dims %mp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mm_b = tt.broadcast %mm_e : tensor<32x1xi1> -> tensor<32x32xi1>
  %mk_b = tt.broadcast %mk_e : tensor<1x32xi1> -> tensor<32x32xi1>
  %mp_b = tt.broadcast %mp_e : tensor<1x32xi1> -> tensor<32x32xi1>
  %load_mask = arith.andi %mm_b, %mk_b : tensor<32x32xi1>
  %store_mask = arith.andi %mm_b, %mp_b : tensor<32x32xi1>
  %im64 = arith.extsi %im : tensor<32xi32> to tensor<32xi64>
  %ik64 = arith.extsi %ik : tensor<32xi32> to tensor<32xi64>
  %im_e = tt.expand_dims %im64 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
  %ik_e = tt.expand_dims %ik64 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
  %im_b = tt.broadcast %im_e : tensor<32x1xi64> -> tensor<32x32xi64>
  %ik_b = tt.broadcast %ik_e : tensor<1x32xi64> -> tensor<32x32xi64>
  %src_ld = tt.splat %source_ld : i64 -> tensor<32x32xi64>
  %base_s = tt.splat %base : i64 -> tensor<32x32xi64>
  %src_col = arith.muli %ik_b, %src_ld : tensor<32x32xi64>
  %src_relative = arith.addi %src_col, %im_b : tensor<32x32xi64>
  %src_offset = arith.addi %src_relative, %base_s : tensor<32x32xi64>
  %ld64 = arith.extsi %ld : i32 to i64
  %dst_ld = tt.splat %ld64 : i64 -> tensor<32x32xi64>
  %dst_col = arith.muli %im_b, %dst_ld : tensor<32x32xi64>
  %dst_offset = arith.addi %dst_col, %ik_b : tensor<32x32xi64>
  %src_s = tt.splat %src : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %dst_s = tt.splat %dst : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %src_p = tt.addptr %src_s, %src_offset : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
  %dst_p = tt.addptr %dst_s, %dst_offset : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi64>
  %zero = arith.constant dense<0.0> : tensor<32x32xf32>
  %value = tt.load %src_p, %load_mask, %zero : tensor<32x32x!tt.ptr<f32>>
  tt.store %dst_p, %value, %store_mask : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

)mlir";
}

static void emitMatmulPackADescriptor(int32_t id, StringRef name,
                                      int32_t imageIndex, llvm::raw_ostream &os,
                                      bool &firstKernel) {
  if (!firstKernel)
    os << ",\n";
  firstKernel = false;
  // Private helper ABI: src, dst, M, K, packed_ld, source_base, source_ld.
  // It is launched only by the runtime, never by a source-level launch.
  os << "    {\n      \"id\": " << id << ",\n"
     << "      \"name\": \"" << name << "\",\n"
     << "      \"kind\": \"matmul_pack_a\",\n"
     << "      \"backend\": \"triton\",\n"
     << "      \"accelerator_target\": \"cuda\",\n"
     << "      \"device_ir_kind\": \"ttir\",\n"
     << "      \"device_image_kind\": \"ptx\",\n"
     << "      \"image_index\": " << imageIndex << ",\n"
     << "      \"ptx_index\": " << imageIndex << ",\n"
     << "      \"image_file\": \"" << name << ".ptx\",\n"
     << "      \"ptx_file\": \"" << name << ".ptx\",\n"
     << "      \"rank\": 2, \"tile\": [32,32,1],\n"
     << "      \"num_warps\": 4, \"threads_per_warp\": 32,\n"
     << "      \"num_ctas\": 1, \"num_stages\": 1,\n"
     << "      \"cuda_threads_per_cta\": 128,\n"
     << "      \"triton_hidden_ptr_args\": 2,\n"
     << "      \"launch_abi_version\": 1,\n"
     << "      \"params\": [\n";
  const char *names[] = {"src", "dst", "m", "k", "ld", "base", "source_ld"};
  const char *types[] = {"ptr<f32>", "ptr<f32>", "i32", "i32",
                         "i32",      "i64",      "i64"};
  for (int i = 0; i < 7; ++i) {
    if (i)
      os << ",\n";
    os << "        {\"slot\": " << i << ", \"name\": \"" << names[i]
       << "\", \"type\": \"" << types[i] << "\", \"role\": \""
       << (i == 0   ? "read"
           : i == 1 ? "write"
                    : "scalar")
       << "\"}";
  }
  os << "\n      ], \"pack\": []\n    }";
}

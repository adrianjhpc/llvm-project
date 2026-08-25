// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null
// RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json

module {
  func.func @iteration_private_scalar(
      %nref: !fir.ref<i32>, %i: !fir.ref<i32>, %tmp: !fir.ref<f32>,
      %alpha: !fir.ref<f32>, %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fnacc.launch tile_sizes = [128] {
      fir.do_loop %iv = %c1 to %n step %c1 : i32 {
        fir.store %iv to %i : !fir.ref<i32>
        %alpha_v = fir.load %alpha : !fir.ref<f32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %product = arith.mulf %alpha_v, %av : f32
        fir.store %product to %tmp : !fir.ref<f32>
        %tmp_v = fir.load %tmp : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>
        %sum = arith.addf %tmp_v, %bv : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %sum to %cp : !fir.ref<f32>
      }
      fnacc.terminator
    } attributes {pack_targets = array<i32>}
    return
  }
}

// The read-only alpha is the only scalar ABI argument. The mutable tmp is
// promoted into the tensor expression and must not become %scalar1.
// TTIR: tt.func @fnacc_kernel_0
// TTIR-SAME: %scalar0: f32
// TTIR-SAME: %n: i32
// TTIR-NOT: %scalar1
// TTIR: %[[ALPHA:.*]] = tt.splat %scalar0 : f32 -> tensor<128xf32>
// TTIR: %[[PRODUCT:.*]] = arith.mulf %[[ALPHA]], %read0v
// TTIR: %[[RESULT:.*]] = arith.addf %[[PRODUCT]], %read1v
// TTIR: tt.store %{{.*}}, %[[RESULT]], %mask

// JSON-COUNT-1: "role": "scalar"
// JSON: "name": "scalar0"

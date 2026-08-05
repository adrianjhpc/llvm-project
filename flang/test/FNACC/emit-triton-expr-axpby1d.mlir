// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null
// RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
// RUN: python3 -m json.tool %t.json > /dev/null

module {
  func.func @axpby1d(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %alpha: !fir.ref<f32>,
      %beta: !fir.ref<f32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fnacc.launch tile_sizes = [128] {
      fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>

        %alpha_v = fir.load %alpha : !fir.ref<f32>
        %beta_v = fir.load %beta : !fir.ref<f32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %mul0 = arith.mulf %alpha_v, %av : f32
        %mul1 = arith.mulf %beta_v, %bv : f32
        %r = arith.addf %mul0, %mul1 : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: %[[NREAD:.*]] = arith.constant 2 : i32
// CHECK: %[[NSCALAR:.*]] = arith.constant 2 : i32
// CHECK: call @__fnacc_launch_f32_v1
// CHECK-SAME: %[[NREAD]], %[[NSCALAR]]

// TTIR: tt.func @fnacc_kernel_0
// TTIR-SAME: %a: !tt.ptr<f32>
// TTIR-SAME: %b: !tt.ptr<f32>
// TTIR-SAME: %c: !tt.ptr<f32>
// TTIR-SAME: %scalar0: f32
// TTIR-SAME: %scalar1: f32
// TTIR-SAME: %n: i32
// TTIR: %[[S0:.*]] = tt.splat %scalar0 : f32 -> tensor<128xf32>
// TTIR: %[[M0:.*]] = arith.mulf %[[S0]], %read0v : tensor<128xf32>
// TTIR: %[[S1:.*]] = tt.splat %scalar1 : f32 -> tensor<128xf32>
// TTIR: %[[M1:.*]] = arith.mulf %[[S1]], %read1v : tensor<128xf32>
// TTIR: %[[R:.*]] = arith.addf %[[M0]], %[[M1]] : tensor<128xf32>
// TTIR: tt.store %{{.*}}, %[[R]], %mask : tensor<128x!tt.ptr<f32>>

// JSON: "cuda_threads_per_cta": 32
// JSON: "triton_hidden_ptr_args": 2
// JSON: "role": "scalar"
// JSON: "name": "scalar0"
// JSON: "role": "scalar"
// JSON: "name": "scalar1"


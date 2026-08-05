// RUN: fir-opt \
// RUN:   --fngpu-assign-kernel-ids \
// RUN:   --fngpu-lower-to-runtime \
// RUN:   %s | FileCheck %s

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

    fngpu.launch tile_sizes = [128] {
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

// CHECK: func.func private @__fngpu_launch_f32_v1
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32
// CHECK-SAME: !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>
// CHECK-SAME: f32, f32, f32
// CHECK-SAME: i32, i32, i32

// CHECK-LABEL: func.func @axpby1d
// CHECK: fir.load %arg2 : !fir.ref<f32>
// CHECK: fir.load %arg3 : !fir.ref<f32>
// CHECK: call @__fngpu_launch_f32_v1
// CHECK-SAME: i32, i32, i32, i32, i32, i32, i32
// CHECK-SAME: !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>
// CHECK-SAME: f32, f32, f32
// CHECK-SAME: i32, i32, i32


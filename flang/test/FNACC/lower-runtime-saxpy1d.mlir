// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @saxpy1d(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %alpha: !fir.ref<f32>,
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

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %mul = arith.mulf %alpha_v, %av : f32

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %r = arith.addf %mul, %bv : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      fnacc.terminator
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK-DAG: func.func private @__fnacc_begin_launch_v2
// CHECK-DAG: func.func private @__fnacc_bind_array_v2
// CHECK-DAG: func.func private @__fnacc_bind_scalar_f32_v2
// CHECK-DAG: func.func private @__fnacc_commit_launch_v2

// CHECK-LABEL: func.func @saxpy1d
// CHECK: call @__fnacc_begin_launch_v2
// CHECK-COUNT-3: call @__fnacc_bind_array_v2
// CHECK: call @__fnacc_bind_scalar_f32_v2
// CHECK: call @__fnacc_commit_launch_v2

// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @dot_reduce(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %sum: !fir.ref<f32>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fnacc.launch tile_sizes = [256] pack(%sum : !fir.ref<f32>) {
      fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>

        %old = fir.load %sum : !fir.ref<f32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %prod = arith.mulf %av, %bv : f32
        %new = arith.addf %old, %prod : f32
        fir.store %new to %sum : !fir.ref<f32>
      }

      fnacc.terminator
    } attributes {
      fnacc.reduction_ops = array<i32: 0>,
      fnacc.reduction_slots = array<i32: 0>,
      pack_targets = array<i32: 0>
    }

    return
  }
}

// CHECK-DAG: func.func private @__fnacc_begin_launch_v2
// CHECK-DAG: func.func private @__fnacc_bind_array_v2
// CHECK-DAG: func.func private @__fnacc_bind_reduction_result_f32_v2
// CHECK-DAG: func.func private @__fnacc_commit_launch_v2

// CHECK-LABEL: func.func @dot_reduce
// CHECK: call @__fnacc_begin_launch_v2
// CHECK-COUNT-2: call @__fnacc_bind_array_v2
// CHECK: call @__fnacc_bind_reduction_result_f32_v2
// CHECK: call @__fnacc_commit_launch_v2
// CHECK-NOT: call @__fnacc_launch_reduce_f32_v2
// CHECK-NOT: fnacc.launch

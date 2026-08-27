// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @kernel1d(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fnacc.launch tile_sizes = [128] pack(%a, %c : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>) {
      fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %r = arith.addf %av, %bv : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      fnacc.terminator
    } attributes {pack_targets = array<i32: 0, 1>}

    return
  }

  func.func @kernel2d(
      %nref: !fir.ref<i32>,
      %mref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %j: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?x?xf32>>,
      %b: !fir.ref<!fir.array<?x?xf32>>,
      %c: !fir.ref<!fir.array<?x?xf32>>) {
    %c1_i32 = arith.constant 1 : i32

    %n = fir.load %nref : !fir.ref<i32>
    %m = fir.load %mref : !fir.ref<i32>

    %nidx = fir.convert %n : (i32) -> index
    %midx = fir.convert %m : (i32) -> index
    %shape = fir.shape %nidx, %midx : (index, index) -> !fir.shape<2>

    fnacc.launch tile_sizes = [16, 16] pack(%a, %c : !fir.ref<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>) {
      fir.do_loop %jv = %c1_i32 to %m step %c1_i32 : i32 {
        fir.store %jv to %j : !fir.ref<i32>

        fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
          fir.store %iv to %i : !fir.ref<i32>

          %i0 = fir.load %i : !fir.ref<i32>
          %idxi0 = fir.convert %i0 : (i32) -> i64
          %j0 = fir.load %j : !fir.ref<i32>
          %idxj0 = fir.convert %j0 : (i32) -> i64
          %ap = fir.array_coor %a(%shape) %idxi0, %idxj0 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          %av = fir.load %ap : !fir.ref<f32>

          %i1 = fir.load %i : !fir.ref<i32>
          %idxi1 = fir.convert %i1 : (i32) -> i64
          %j1 = fir.load %j : !fir.ref<i32>
          %idxj1 = fir.convert %j1 : (i32) -> i64
          %bp = fir.array_coor %b(%shape) %idxi1, %idxj1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          %bv = fir.load %bp : !fir.ref<f32>

          %r = arith.mulf %av, %bv : f32

          %i2 = fir.load %i : !fir.ref<i32>
          %idxi2 = fir.convert %i2 : (i32) -> i64
          %j2 = fir.load %j : !fir.ref<i32>
          %idxj2 = fir.convert %j2 : (i32) -> i64
          %cp = fir.array_coor %c(%shape) %idxi2, %idxj2 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          fir.store %r to %cp : !fir.ref<f32>
        }
      }

      fnacc.terminator
    } attributes {pack_targets = array<i32: 0, 1>}

    return
  }
}

// CHECK-DAG: func.func private @__fnacc_begin_launch_v2
// CHECK-DAG: func.func private @__fnacc_bind_array_v2
// CHECK-DAG: func.func private @__fnacc_commit_launch_v2

// CHECK-LABEL: func.func @kernel1d
// CHECK: call @__fnacc_begin_launch_v2
// CHECK-COUNT-3: call @__fnacc_bind_array_v2
// CHECK: call @__fnacc_commit_launch_v2

// CHECK-LABEL: func.func @kernel2d
// CHECK: call @__fnacc_begin_launch_v2
// CHECK-COUNT-3: call @__fnacc_bind_array_v2
// CHECK: call @__fnacc_commit_launch_v2

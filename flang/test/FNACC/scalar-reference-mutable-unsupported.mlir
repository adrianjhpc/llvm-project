// RUN: not fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @loop_carried_scalar(
      %nref: !fir.ref<i32>, %i: !fir.ref<i32>, %tmp: !fir.ref<f32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fnacc.launch tile_sizes = [128] {
      fir.do_loop %iv = %c1 to %n step %c1 : i32 {
        fir.store %iv to %i : !fir.ref<i32>
        %old_tmp = fir.load %tmp : !fir.ref<f32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>
        %next_tmp = arith.addf %old_tmp, %av : f32
        fir.store %next_tmp to %tmp : !fir.ref<f32>
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

// CHECK: error: FNACC cannot plan launch:
// CHECK-SAME: expression-tree failure: mutable scalar reference is neither iteration-private nor a reduction

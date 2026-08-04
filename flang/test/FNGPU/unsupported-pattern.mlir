// RUN: fir-opt \
// RUN:   --fngpu-assign-kernel-ids \
// RUN:   --fngpu-lower-to-runtime \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @unsupported_three_reads(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %d: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    fngpu.launch tile_sizes = [128] {
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

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %dp = fir.array_coor %d(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %dv = fir.load %dp : !fir.ref<f32>

        %tmp = arith.addf %av, %bv : f32
        %r = arith.addf %tmp, %dv : f32

        %i3 = fir.load %i : !fir.ref<i32>
        %idx3 = fir.convert %i3 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx3 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: warning: FNGPU runtime lowering skipped launch:
// CHECK-SAME: kernel expected exactly two read arrays


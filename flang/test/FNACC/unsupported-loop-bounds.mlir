// RUN: fir-opt \
// RUN:   --fngpu-assign-kernel-ids \
// RUN:   --fngpu-lower-to-runtime \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @bad_lower_bound(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>

    fngpu.launch tile_sizes = [128] {
      fir.do_loop %iv = %c0_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }

  func.func @bad_step(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %n = fir.load %nref : !fir.ref<i32>

    fngpu.launch tile_sizes = [128] {
      fir.do_loop %iv = %c1_i32 to %n step %c2_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: warning: FNGPU runtime lowering skipped launch:
// CHECK-SAME: 1-D loop lower bound must be constant 1

// CHECK: warning: FNGPU runtime lowering skipped launch:
// CHECK-SAME: 1-D loop step must be constant 1


// RUN: fir-opt --fnacc-assign-kernel-ids %s | FileCheck %s

module {
  func.func @assign_ids() {
    fnacc.launch tile_sizes = [] {
      fnacc.terminator
    } attributes {fnacc.kernel_id = 0 : i32, pack_targets = array<i32>}

    fnacc.launch tile_sizes = [16, 16] {
      fnacc.terminator
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: fnacc.launch tile_sizes = []
// CHECK: attributes
// CHECK-SAME: fnacc.kernel_id = 0 : i32
// CHECK-SAME: fnacc.kernel_name = "fnacc_kernel_0"
// CHECK-SAME: pack_targets = array<i32>

// CHECK: fnacc.launch tile_sizes = [16, 16]
// CHECK: attributes
// CHECK-SAME: fnacc.kernel_id = 1 : i32
// CHECK-SAME: fnacc.kernel_name = "fnacc_kernel_1"
// CHECK-SAME: pack_targets = array<i32>


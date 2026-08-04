// RUN: fir-opt --fngpu-assign-kernel-ids %s | FileCheck %s

module {
  func.func @assign_ids() {
    fngpu.launch tile_sizes = [] {
      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    fngpu.launch tile_sizes = [16, 16] {
      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: fngpu.launch tile_sizes = []
// CHECK: attributes
// CHECK-SAME: fngpu.kernel_id = 0 : i32
// CHECK-SAME: fngpu.kernel_name = "fngpu_kernel_0"
// CHECK-SAME: pack_targets = array<i32>

// CHECK: fngpu.launch tile_sizes = [16, 16]
// CHECK: attributes
// CHECK-SAME: fngpu.kernel_id = 1 : i32
// CHECK-SAME: fngpu.kernel_name = "fngpu_kernel_1"
// CHECK-SAME: pack_targets = array<i32>


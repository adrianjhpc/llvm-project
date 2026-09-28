// RUN: fir-opt --TileOffload-assign-kernel-ids %s | FileCheck %s

module {
  func.func @assign_ids() {
    TileOffload.launch tile_sizes = [] {
      TileOffload.terminator
    } attributes {TileOffload.kernel_id = 0 : i32, pack_targets = array<i32>}

    TileOffload.launch tile_sizes = [16, 16] {
      TileOffload.terminator
    } attributes {pack_targets = array<i32>}

    return
  }
}

// CHECK: TileOffload.launch tile_sizes = []
// CHECK: attributes
// CHECK-SAME: TileOffload.kernel_id = 0 : i32
// CHECK-SAME: TileOffload.kernel_name = "tileoff_kernel_0"
// CHECK-SAME: pack_targets = array<i32>

// CHECK: TileOffload.launch tile_sizes = [16, 16]
// CHECK: attributes
// CHECK-SAME: TileOffload.kernel_id = 1 : i32
// CHECK-SAME: TileOffload.kernel_name = "tileoff_kernel_1"
// CHECK-SAME: pack_targets = array<i32>


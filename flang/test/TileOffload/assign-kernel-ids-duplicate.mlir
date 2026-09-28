// RUN: not fir-opt --TileOffload-assign-kernel-ids %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @duplicate_ids() {
    TileOffload.launch tile_sizes = [64] {
      TileOffload.terminator
    } attributes {TileOffload.kernel_id = 7 : i32, pack_targets = array<i32>}

    TileOffload.launch tile_sizes = [64] {
      TileOffload.terminator
    } attributes {TileOffload.kernel_id = 7 : i32, pack_targets = array<i32>}

    return
  }
}

// CHECK: error: duplicate TileOffload kernel id 7

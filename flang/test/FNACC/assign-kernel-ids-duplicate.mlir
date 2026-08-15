// RUN: not fir-opt --fnacc-assign-kernel-ids %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @duplicate_ids() {
    fnacc.launch tile_sizes = [64] {
      "fir.end"() : () -> ()
    } attributes {fnacc.kernel_id = 7 : i32, pack_targets = array<i32>}

    fnacc.launch tile_sizes = [64] {
      "fir.end"() : () -> ()
    } attributes {fnacc.kernel_id = 7 : i32, pack_targets = array<i32>}

    return
  }
}

// CHECK: error: duplicate FNACC kernel id 7

// RUN: not fir-opt %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @bad_pack_mismatch(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>) {
    fngpu.launch tile_sizes = [128] pack(%a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>) {
      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32: 1>}

    return
  }
}

// CHECK: expected pack_targets to have exactly one entry per pack var


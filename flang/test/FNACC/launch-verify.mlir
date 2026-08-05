// RUN: not fir-opt %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @bad_pack_target(
      %a: !fir.ref<!fir.array<?xf32>>) {
    fngpu.launch tile_sizes = [] pack(%a : !fir.ref<!fir.array<?xf32>>) {
      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32: 7>}

    return
  }
}

// CHECK: pack target must be 0 (host) or 1 (device)


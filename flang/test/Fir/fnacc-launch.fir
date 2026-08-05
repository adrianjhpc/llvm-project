// RUN: fir-opt %s | FileCheck %s

module {
  func.func @test_fnacc_launch(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>) {
    fnacc.launch tile_sizes = [64] pack(%a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>) {
      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32: 0, 1>}

    return
  }
}

// CHECK: fnacc.launch tile_sizes = [64]
// CHECK-SAME: pack(%{{.*}}, %{{.*}} : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
// CHECK: attributes
// CHECK-SAME: pack_targets = array<i32: 0, 1>


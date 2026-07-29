// RUN: fir-opt %s | fir-opt | FileCheck %s
// CHECK: fngpu.launch tile_sizes = [64] pack(%{{.*}}, %{{.*}} : {{.*}}) {{.*}}array<i32: 0, 1>
func.func @t(%a: !fir.ref<!fir.array<?xf32>>, %c: !fir.ref<!fir.array<?xf32>>) {
  fngpu.launch tile_sizes = [64]
      pack(%a, %c : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
      {pack_targets = array<i32: 0, 1>} {
    fir.end
  }
  return
}


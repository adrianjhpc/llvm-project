// RUN: fir-opt --split-input-file --verify-diagnostics %s -o /dev/null
module {
  func.func @bad_value() {
    // expected-error @+1 {{matmul_precision must be the string ieee, tf32, or tf32x3}}
    "TileOffload.launch"() ({ "TileOffload.terminator"() : () -> () }) {tile_sizes = array<i64: 16, 16>, pack_targets = array<i32>, TileOffload.matmul_precision = "bad"} : () -> ()
    return
  }
}
// -----
module {
  func.func @bad_type() {
    // expected-error @+1 {{matmul_precision must be the string ieee, tf32, or tf32x3}}
    "TileOffload.launch"() ({ "TileOffload.terminator"() : () -> () }) {tile_sizes = array<i64: 16, 16>, pack_targets = array<i32>, TileOffload.matmul_precision = 0 : i32} : () -> ()
    return
  }
}

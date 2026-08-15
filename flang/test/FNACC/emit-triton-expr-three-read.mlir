// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null
// RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
// RUN: python3 -m json.tool %t.json > /dev/null

module {
  func.func @expr_three_read(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %d: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    // c(i) = a(i) + b(i) + d(i)
    //
    // This used to be unsupported when FNACC required exactly two read arrays.
    // It should now lower as a generic Expr1D kernel with three read arrays.
    fnacc.launch tile_sizes = [128] {
      fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %ap = fir.array_coor %a(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %av = fir.load %ap : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %dp = fir.array_coor %d(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %dv = fir.load %dp : !fir.ref<f32>

        %tmp = arith.addf %av, %bv : f32
        %r = arith.addf %tmp, %dv : f32

        %i3 = fir.load %i : !fir.ref<i32>
        %idx3 = fir.convert %i3 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx3 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32>}

    return
  }
}

// TTIR: module attributes
// TTIR: tt.func @fnacc_kernel_0
// TTIR-SAME: %read0: !tt.ptr<f32>
// TTIR-SAME: %read1: !tt.ptr<f32>
// TTIR-SAME: %read2: !tt.ptr<f32>
// TTIR-SAME: %c: !tt.ptr<f32>
// TTIR-SAME: %n: i32

// TTIR: %[[READ0P:.*]] = tt.splat %read0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// TTIR: %[[READ0O:.*]] = tt.addptr %[[READ0P]], %offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// TTIR: %[[READ0V:.*]] = tt.load %[[READ0O]], %mask : tensor<128x!tt.ptr<f32>>

// TTIR: %[[READ1P:.*]] = tt.splat %read1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// TTIR: %[[READ1O:.*]] = tt.addptr %[[READ1P]], %offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// TTIR: %[[READ1V:.*]] = tt.load %[[READ1O]], %mask : tensor<128x!tt.ptr<f32>>

// TTIR: %[[READ2P:.*]] = tt.splat %read2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// TTIR: %[[READ2O:.*]] = tt.addptr %[[READ2P]], %offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// TTIR: %[[READ2V:.*]] = tt.load %[[READ2O]], %mask : tensor<128x!tt.ptr<f32>>

// TTIR: %[[ADD0:.*]] = arith.addf %[[READ0V]], %[[READ1V]] : tensor<128xf32>
// TTIR: %[[ADD1:.*]] = arith.addf %[[ADD0]], %[[READ2V]] : tensor<128xf32>
// TTIR: tt.store %{{.*}}, %[[ADD1]], %mask : tensor<128x!tt.ptr<f32>>
// TTIR: tt.return
// TTIR-NOT: fastmath
// TTIR-NOT: math.fma

// JSON: "fnacc_schema_version": 1
// JSON: "id": 0
// JSON: "name": "fnacc_kernel_0"
// JSON: "kind": "expr1d"
// JSON: "rank": 1
// JSON: "tile": [128, 1, 1]

// JSON: "params": [
// JSON: "slot": 0
// JSON-SAME: "role": "read"
// JSON-SAME: "name": "read0"

// JSON: "slot": 1
// JSON-SAME: "role": "read"
// JSON-SAME: "name": "read1"

// JSON: "slot": 2
// JSON-SAME: "role": "read"
// JSON-SAME: "name": "read2"

// JSON: "slot": 3
// JSON-SAME: "role": "write"
// JSON-SAME: "name": "write"

// JSON: "role": "extent_x"
// JSON-SAME: "name": "extent_x"


// RUN: fir-opt \
// RUN:   --fnacc-assign-kernel-ids \
// RUN:   --fnacc-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null
// RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
// RUN: python3 -m json.tool %t.json > /dev/null

module {
  func.func @kernel1d(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

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

        %r = arith.subf %av, %bv : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp : !fir.ref<f32>
      }

      fnacc.terminator
    } attributes {pack_targets = array<i32>}

    return
  }

  func.func @kernel2d(
      %nref: !fir.ref<i32>,
      %mref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %j: !fir.ref<i32>,
      %a: !fir.ref<!fir.array<?x?xf32>>,
      %b: !fir.ref<!fir.array<?x?xf32>>,
      %c: !fir.ref<!fir.array<?x?xf32>>) {
    %c1_i32 = arith.constant 1 : i32

    %n = fir.load %nref : !fir.ref<i32>
    %m = fir.load %mref : !fir.ref<i32>

    %nidx = fir.convert %n : (i32) -> index
    %midx = fir.convert %m : (i32) -> index
    %shape = fir.shape %nidx, %midx : (index, index) -> !fir.shape<2>

    fnacc.launch tile_sizes = [16, 16] {
      fir.do_loop %jv = %c1_i32 to %m step %c1_i32 : i32 {
        fir.store %jv to %j : !fir.ref<i32>

        fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
          fir.store %iv to %i : !fir.ref<i32>

          %i0 = fir.load %i : !fir.ref<i32>
          %idxi0 = fir.convert %i0 : (i32) -> i64
          %j0 = fir.load %j : !fir.ref<i32>
          %idxj0 = fir.convert %j0 : (i32) -> i64
          %ap = fir.array_coor %a(%shape) %idxi0, %idxj0 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          %av = fir.load %ap : !fir.ref<f32>

          %i1 = fir.load %i : !fir.ref<i32>
          %idxi1 = fir.convert %i1 : (i32) -> i64
          %j1 = fir.load %j : !fir.ref<i32>
          %idxj1 = fir.convert %j1 : (i32) -> i64
          %bp = fir.array_coor %b(%shape) %idxi1, %idxj1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          %bv = fir.load %bp : !fir.ref<f32>

          %r = arith.divf %av, %bv : f32

          %i2 = fir.load %i : !fir.ref<i32>
          %idxi2 = fir.convert %i2 : (i32) -> i64
          %j2 = fir.load %j : !fir.ref<i32>
          %idxj2 = fir.convert %j2 : (i32) -> i64
          %cp = fir.array_coor %c(%shape) %idxi2, %idxj2 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
          fir.store %r to %cp : !fir.ref<f32>
        }
      }

      fnacc.terminator
    } attributes {pack_targets = array<i32>}

    return
  }
}

// TTIR: module attributes
// TTIR: tt.func @fnacc_kernel_0
// TTIR-SAME: %n: i32
// TTIR: arith.subf
// TTIR: tt.func @fnacc_kernel_1
// TTIR-SAME: %n: i32, %m: i32
// TTIR: tt.get_program_id y
// TTIR: arith.divf

// JSON: "id": 0
// JSON: "name": "fnacc_kernel_0"
// JSON: "rank": 1
// JSON: "tile": [128, 1, 1]

// JSON: "id": 1
// JSON: "name": "fnacc_kernel_1"
// JSON: "rank": 2
// JSON: "tile": [16, 16, 1]
// JSON: "grid": ["cdiv(extent_x, tile_x)", "cdiv(extent_y, tile_y)", "1"]


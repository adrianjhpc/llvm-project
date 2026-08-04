// RUN: fir-opt \
// RUN:   --fngpu-assign-kernel-ids \
// RUN:   --fngpu-lower-to-triton="ttir-output=%t.ttir json-output=%t.json" \
// RUN:   %s -o /dev/null
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
// RUN: python3 -m json.tool %t.json > /dev/null

module {
  func.func @pack_inplace(
      %nref: !fir.ref<i32>,
      %i: !fir.ref<i32>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %c: !fir.ref<!fir.array<?xf32>>) {
    %c1_i32 = arith.constant 1 : i32
    %n = fir.load %nref : !fir.ref<i32>
    %nidx = fir.convert %n : (i32) -> index
    %shape = fir.shape %nidx : (index) -> !fir.shape<1>

    // c is both a read array and the write array:
    //
    //   c(i) = c(i) + b(i)
    //
    // So pack(%c) should become two pack entries:
    //
    //   slot 0 or 1, depending read order, for the read use of c
    //   slot 2 for the write use of c
    fngpu.launch tile_sizes = [128] pack(%c : !fir.ref<!fir.array<?xf32>>) {
      fir.do_loop %iv = %c1_i32 to %n step %c1_i32 : i32 {
        fir.store %iv to %i : !fir.ref<i32>

        %i0 = fir.load %i : !fir.ref<i32>
        %idx0 = fir.convert %i0 : (i32) -> i64
        %cp_read = fir.array_coor %c(%shape) %idx0 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %cv = fir.load %cp_read : !fir.ref<f32>

        %i1 = fir.load %i : !fir.ref<i32>
        %idx1 = fir.convert %i1 : (i32) -> i64
        %bp = fir.array_coor %b(%shape) %idx1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        %bv = fir.load %bp : !fir.ref<f32>

        %r = arith.addf %cv, %bv : f32

        %i2 = fir.load %i : !fir.ref<i32>
        %idx2 = fir.convert %i2 : (i32) -> i64
        %cp_write = fir.array_coor %c(%shape) %idx2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
        fir.store %r to %cp_write : !fir.ref<f32>
      }

      "fir.end"() : () -> ()
    } attributes {pack_targets = array<i32: 1>}

    return
  }
}

// JSON: "pack": [
// JSON: "kernel_arg_slot": 0
// JSON: "target": 1
// JSON: "target_name": "device"
// JSON: "kernel_arg_slot": 2
// JSON: "target": 1
// JSON: "target_name": "device"


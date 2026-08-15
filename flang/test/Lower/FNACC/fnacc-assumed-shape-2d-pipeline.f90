! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine matrix_add_assumed_shape(a, b, c)
  real :: a(:, :), b(:, :), c(:, :)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, size(c, 2)
    do i = 1, size(c, 1)
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_launch_f32_v1
! HOST-DAG: func.func private @__fnacc_validate_contiguous_desc

! HOST-LABEL: func.func @_QPmatrix_add_assumed_shape
! HOST: fir.box_dims
! HOST: fir.box_dims
! HOST-COUNT-3: call @__fnacc_validate_contiguous_desc
! HOST: fir.box_addr
! HOST: fir.box_addr
! HOST: fir.box_addr
! HOST: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: tt.get_program_id y
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "kind": "binary"
! JSON: "rank": 2
! JSON: "tile": [16, 16, 1]
! JSON: "grid": ["cdiv(extent_x, tile_x)", "cdiv(extent_y, tile_y)", "1"]


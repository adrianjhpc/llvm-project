! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine matrix_add_assumed_shape(a, b, c)
  real :: a(:, :), b(:, :), c(:, :)
  integer :: i, j

  !$tileoff parallel tile(16, 16)
  do j = 1, size(c, 2)
    do i = 1, size(c, 1)
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__tileoff_validate_contiguous_desc
! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_bind_array_v2
! HOST-DAG: func.func private @__tileoff_commit_launch_v2

! HOST-LABEL: func.func @_QPmatrix_add_assumed_shape
! HOST-COUNT-3: call @__tileoff_validate_contiguous_desc
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-3: call @__tileoff_bind_array_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST-NOT: TileOffload.launch

! TTIR: tt.func @tileoff_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: tt.get_program_id y
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "tileoff_kernel_0"
! JSON: "kind": "binary"
! JSON: "rank": 2
! JSON: "tile": [16, 16, 1]
! JSON: "grid": ["cdiv(extent_x, tile_x)", "cdiv(extent_y, tile_y)", "1"]

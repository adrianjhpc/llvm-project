! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine vector_add_assumed_shape(a, b, c)
  real :: a(:), b(:), c(:)
  integer :: i
  integer :: n

  n = size(c)

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_validate_contiguous_desc
! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2

! HOST-LABEL: func.func @_QPvector_add_assumed_shape
! HOST-COUNT-3: call @__fnacc_validate_contiguous_desc
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-3: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "kind": "binary"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]

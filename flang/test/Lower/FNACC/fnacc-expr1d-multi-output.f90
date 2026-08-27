! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine multi_output_expr1d(n, a, b, sum, difference)
  integer :: n
  real :: a(n), b(n), sum(n), difference(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    sum(i) = a(i) + b(i)
    difference(i) = a(i) - b(i)
  end do
end subroutine

! HOST-LABEL: func.func @_QPmulti_output_expr1d
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-4: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2

! TTIR: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f32>
! TTIR-SAME: %array1: !tt.ptr<f32>
! TTIR-SAME: %array2: !tt.ptr<f32>
! TTIR-SAME: %array3: !tt.ptr<f32>
! TTIR-SAME: %extent_x: i32
! TTIR: arith.addf
! TTIR: tt.store
! TTIR: arith.subf
! TTIR: tt.store

! JSON: "kind": "multi_expr1d"
! JSON: "rank": 1
! JSON: "launch_abi_version": 2
! JSON: "array_count": 4
! JSON: "scalar_count": 0
! JSON: "output_count": 2

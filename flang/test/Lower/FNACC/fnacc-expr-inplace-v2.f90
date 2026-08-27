! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine inplace_v2(n, a, alpha)
  integer :: n
  real :: a(n), alpha
  integer :: i

  !$fnacc parallel tile(128) pack(a:device)
  do i = 1, n
    a(i) = alpha * a(i)
  end do
end subroutine

! HOST-LABEL: func.func @_QPinplace_v2
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-1: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_bind_scalar_f32_v2
! HOST: call @__fnacc_commit_launch_v2

! JSON: "kind": "expr1d"
! JSON: "array_count": 1
! JSON: "scalar_count": 1
! JSON: "role": "read"
! JSON-SAME: "array_index": 0
! JSON: "role": "write"
! JSON-SAME: "array_index": 0
! JSON: "kernel_arg_slot": 0
! JSON: "kernel_arg_slot": 1

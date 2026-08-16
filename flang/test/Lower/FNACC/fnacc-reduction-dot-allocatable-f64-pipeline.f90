! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_dot_reduce_allocatable_f64(n, a, b, sum)
  use, intrinsic :: iso_fortran_env, only : real64
  integer, intent(in) :: n
  real(kind=real64), allocatable, intent(in) :: a(:), b(:)
  real(kind=real64), intent(out) :: sum
  integer :: i

  sum = 0.0_real64

  !$fnacc parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i) * b(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_launch_reduce_f64_v1

! HOST-LABEL: func.func @_QPfnacc_dot_reduce_allocatable_f64
! HOST: call @__fnacc_launch_reduce_f64_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %b: !tt.ptr<f64>
! TTIR-SAME: %partials: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR: arith.mulf
! TTIR-SAME: f64
! TTIR: tt.reduce
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "kind": "reduction_dot1d"
! JSON-DAG: "type": "ptr<f64>"
! JSON-DAG: "role": "partials"

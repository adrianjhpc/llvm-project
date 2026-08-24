! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine fnacc_expr_too_many_scalars(n, a, c, s0, s1, s2, s3)
  integer :: n
  real :: a(n), c(n)
  real :: s0, s1, s2, s3
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + s0 + s1 + s2 + s3
  end do
end subroutine

! CHECK: error: FNACC Triton cannot emit launch:
! CHECK-SAME: expression tree supports at most three scalar


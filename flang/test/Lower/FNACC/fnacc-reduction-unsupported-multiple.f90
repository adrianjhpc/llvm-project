! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine fnacc_bad_multi_reduce(n, a, sum1, sum2)
  integer :: n
  real :: a(n)
  real :: sum1, sum2
  integer :: i

  sum1 = 0.0
  sum2 = 0.0

  !$fnacc parallel tile(256) reduction(+:sum1, +:sum2)
  do i = 1, n
    sum1 = sum1 + a(i)
    sum2 = sum2 + a(i)
  end do
end subroutine

! CHECK: warning: FNACC
! CHECK-SAME: reduction recognition requires exactly one reduction scalar


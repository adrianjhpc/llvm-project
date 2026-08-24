! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine bad_lower_bound(n, a, b, c)
  integer :: n
  real :: a(0:n), b(0:n), c(0:n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 0, n
    c(i) = a(i) + b(i)
  end do
end subroutine

subroutine bad_step(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n, 2
    c(i) = a(i) + b(i)
  end do
end subroutine

! CHECK: error: FNACC cannot plan launch:
! CHECK-SAME: loop lower bound must be constant 1

! CHECK: error: FNACC cannot plan launch:
! CHECK-SAME: loop step must be constant 1


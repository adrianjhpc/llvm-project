! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine bad_2d_step(n, m, a, b, c)
  integer :: n, m
  real :: a(n, m), b(n, m), c(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n, 2
      c(i, j) = a(i, j) + b(i, j)
    end do
  end do
end subroutine

! CHECK: error: FNACC cannot plan launch:
! CHECK-SAME: loop step must be constant 1

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine fnacc_expr_mixed_types(n, a, b, c)
  integer :: n
  real :: a(n), c(n)
  real(8) :: b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + real(b(i))
  end do
end subroutine

! CHECK: error: FNACC Triton cannot emit launch:
! CHECK-SAME: all read/write arrays must have the same real element type


! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt \
! RUN:   --tileoff-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine tileoff_expr_unsupported_intrinsic(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$tileoff parallel tile(128)
  do i = 1, n
    c(i) = atan2(a(i), b(i))
  end do
end subroutine

! CHECK: error: TileOffload cannot plan launch:
! CHECK-SAME: unsupported operation


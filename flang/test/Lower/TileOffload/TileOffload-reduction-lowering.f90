! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_tileoff_reduction_lowering(n, a, b, sum)
  integer :: n
  real :: a(n), b(n)
  real :: sum
  integer :: i

  !$tileoff parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i) * b(i)
  end do
end subroutine

! CHECK: TileOffload.launch
! CHECK-SAME: tile_sizes = [256]
! CHECK: attributes
! CHECK-SAME: TileOffload.reduction_ops = array<i32: 0>
! CHECK-SAME: TileOffload.reduction_slots = array<i32:


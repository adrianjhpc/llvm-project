! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine pack_directive(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128) pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine

! CHECK: fnacc.launch
! CHECK-SAME: tile_sizes = [128]
! CHECK: pack(
! CHECK: pack_targets = array<i32: 1, 1>


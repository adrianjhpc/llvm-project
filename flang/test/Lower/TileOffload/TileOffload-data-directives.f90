! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_tileoff_data(n, a, c)
  integer :: n
  real :: a(n), c(n)

  !$tileoff update host(c)
  !$tileoff update device(a)
  !$tileoff release(a, c)
  !$tileoff release all
end subroutine

! CHECK: TileOffload.update_host
! CHECK: TileOffload.update_device
! CHECK: TileOffload.release
! CHECK: TileOffload.release_all


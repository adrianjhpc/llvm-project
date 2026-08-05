! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_fngpu_data(n, a, c)
  integer :: n
  real :: a(n), c(n)

  !$fngpu update host(c)
  !$fngpu update device(a)
  !$fngpu release(a, c)
  !$fngpu release all
end subroutine

! CHECK: fngpu.update_host
! CHECK: fngpu.update_device
! CHECK: fngpu.release
! CHECK: fngpu.release_all


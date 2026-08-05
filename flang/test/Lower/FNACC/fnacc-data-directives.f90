! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_fnacc_data(n, a, c)
  integer :: n
  real :: a(n), c(n)

  !$fnacc update host(c)
  !$fnacc update device(a)
  !$fnacc release(a, c)
  !$fnacc release all
end subroutine

! CHECK: fnacc.update_host
! CHECK: fnacc.update_device
! CHECK: fnacc.release
! CHECK: fnacc.release_all


! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_fnacc_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

! CHECK: fnacc.copyin
! CHECK: fnacc.copyin
! CHECK: fnacc.create
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.delete
! CHECK: fnacc.delete


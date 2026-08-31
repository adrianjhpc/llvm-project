! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_fnacc_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a) create(c)
  !$fnacc enter data copyin(a, b)
  !$fnacc exit data copyout(a, b) delete(a, b)
  !$fnacc exit data copyout(a, c) delete(a, c)
end subroutine

! CHECK: fnacc.data_region_enter
! CHECK: fnacc.copyin
! CHECK: fnacc.create
! CHECK: fnacc.data_region_enter
! CHECK: fnacc.copyin
! CHECK: fnacc.copyin
! CHECK: fnacc.copyout
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.delete
! CHECK: fnacc.data_region_exit
! CHECK: fnacc.copyout
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.delete
! CHECK: fnacc.data_region_exit

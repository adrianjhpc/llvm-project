! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_fnacc_enter_exit_data_runtime(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_bytes
! HOST-DAG: func.func private @__fnacc_create_bytes
! HOST-DAG: func.func private @__fnacc_update_host_bytes
! HOST-DAG: func.func private @__fnacc_release

! HOST-LABEL: func.func @_QPtest_fnacc_enter_exit_data_runtime
! HOST: call @__fnacc_update_device_bytes
! HOST: call @__fnacc_update_device_bytes
! HOST: call @__fnacc_create_bytes
! HOST: call @__fnacc_update_host_bytes
! HOST: call @__fnacc_release
! HOST: call @__fnacc_release
! HOST: call @__fnacc_release

! HOST-NOT: fnacc.copyin
! HOST-NOT: fnacc.create
! HOST-NOT: fnacc.copyout
! HOST-NOT: fnacc.delete


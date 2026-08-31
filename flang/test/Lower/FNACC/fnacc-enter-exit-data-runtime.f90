! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_fnacc_enter_exit_data_runtime(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a) create(c)
  !$fnacc enter data copyin(a, b)
  !$fnacc exit data copyout(a, b) delete(a, b)
  !$fnacc exit data copyout(a, c) delete(a, c)
end subroutine

! HOST-DAG: func.func private @__fnacc_enter_data_region
! HOST-DAG: func.func private @__fnacc_data_copyin_bytes
! HOST-DAG: func.func private @__fnacc_data_create_bytes
! HOST-DAG: func.func private @__fnacc_data_copyout_bytes
! HOST-DAG: func.func private @__fnacc_data_delete
! HOST-DAG: func.func private @__fnacc_exit_data_region

! HOST-LABEL: func.func @_QPtest_fnacc_enter_exit_data_runtime
! HOST: call @__fnacc_enter_data_region
! HOST: call @__fnacc_data_copyin_bytes
! HOST: call @__fnacc_data_create_bytes
! HOST: call @__fnacc_enter_data_region
! HOST: call @__fnacc_data_copyin_bytes
! HOST: call @__fnacc_data_copyin_bytes
! HOST: call @__fnacc_data_copyout_bytes
! HOST: call @__fnacc_data_copyout_bytes
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_exit_data_region
! HOST: call @__fnacc_data_copyout_bytes
! HOST: call @__fnacc_data_copyout_bytes
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_exit_data_region

! HOST-NOT: fnacc.copyin
! HOST-NOT: fnacc.create
! HOST-NOT: fnacc.copyout
! HOST-NOT: fnacc.delete

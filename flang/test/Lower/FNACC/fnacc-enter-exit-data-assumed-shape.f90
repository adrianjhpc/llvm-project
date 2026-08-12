! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_fnacc_enter_exit_data_assumed_shape(a, b, c)
  real :: a(:), b(:), c(:)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_desc
! HOST-DAG: func.func private @__fnacc_create_desc
! HOST-DAG: func.func private @__fnacc_update_host_desc
! HOST-DAG: func.func private @__fnacc_release_desc

! HOST-LABEL: func.func @_QPtest_fnacc_enter_exit_data_assumed_shape
! HOST: call @__fnacc_update_device_desc
! HOST: call @__fnacc_update_device_desc
! HOST: call @__fnacc_create_desc
! HOST: call @__fnacc_update_host_desc
! HOST: call @__fnacc_release_desc
! HOST: call @__fnacc_release_desc
! HOST: call @__fnacc_release_desc


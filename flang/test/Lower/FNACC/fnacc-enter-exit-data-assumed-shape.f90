! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_fnacc_enter_exit_data_assumed_shape(a, b, c)
  real :: a(:), b(:), c(:)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

! HOST-DAG: func.func private @__fnacc_enter_data_region
! HOST-DAG: func.func private @__fnacc_data_copyin_desc
! HOST-DAG: func.func private @__fnacc_data_create_desc
! HOST-DAG: func.func private @__fnacc_data_copyout_desc
! HOST-DAG: func.func private @__fnacc_data_delete
! HOST-DAG: func.func private @__fnacc_exit_data_region

! HOST-LABEL: func.func @_QPtest_fnacc_enter_exit_data_assumed_shape
! HOST: call @__fnacc_enter_data_region
! HOST: call @__fnacc_data_copyin_desc
! HOST: call @__fnacc_data_copyin_desc
! HOST: call @__fnacc_data_create_desc
! HOST: call @__fnacc_data_copyout_desc
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_exit_data_region

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_tileoff_enter_exit_data_assumed_shape(a, b, c)
  real :: a(:), b(:), c(:)

  !$tileoff enter data copyin(a, b) create(c)
  !$tileoff exit data copyout(c) delete(a, b, c)
end subroutine

! HOST-DAG: func.func private @__tileoff_enter_data_region
! HOST-DAG: func.func private @__tileoff_data_copyin_desc
! HOST-DAG: func.func private @__tileoff_data_create_desc
! HOST-DAG: func.func private @__tileoff_data_copyout_desc
! HOST-DAG: func.func private @__tileoff_data_delete
! HOST-DAG: func.func private @__tileoff_exit_data_region

! HOST-LABEL: func.func @_QPtest_tileoff_enter_exit_data_assumed_shape
! HOST: call @__tileoff_enter_data_region
! HOST: call @__tileoff_data_copyin_desc
! HOST: call @__tileoff_data_copyin_desc
! HOST: call @__tileoff_data_create_desc
! HOST: call @__tileoff_data_copyout_desc
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_exit_data_region

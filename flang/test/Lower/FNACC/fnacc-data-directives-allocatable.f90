! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_allocatable_f64(a)
  use, intrinsic :: iso_fortran_env, only : real64
  real(kind=real64), allocatable :: a(:)

  !$fnacc enter data create(a)
  !$fnacc update device(a)
  !$fnacc update host(a)
  !$fnacc exit data delete(a)
end subroutine

! HOST-DAG: func.func private @__fnacc_enter_data_region
! HOST-DAG: func.func private @__fnacc_data_create_desc
! HOST-DAG: func.func private @__fnacc_update_device_desc
! HOST-DAG: func.func private @__fnacc_update_host_desc
! HOST-DAG: func.func private @__fnacc_data_delete
! HOST-DAG: func.func private @__fnacc_exit_data_region

! HOST-LABEL: func.func @_QPdata_allocatable_f64

! HOST: call @__fnacc_enter_data_region
! HOST: %[[CREATE_BOX:.*]] = fir.load {{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
! HOST: fir.box_addr %[[CREATE_BOX]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
! HOST: fir.box_dims %[[CREATE_BOX]]
! HOST: call @__fnacc_data_create_desc

! HOST: %[[DEVICE_BOX:.*]] = fir.load {{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
! HOST: fir.box_addr %[[DEVICE_BOX]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
! HOST: fir.box_dims %[[DEVICE_BOX]]
! HOST: call @__fnacc_update_device_desc

! HOST: %[[HOST_BOX:.*]] = fir.load {{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
! HOST: fir.box_addr %[[HOST_BOX]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
! HOST: fir.box_dims %[[HOST_BOX]]
! HOST: call @__fnacc_update_host_desc

! HOST: %[[DELETE_BOX:.*]] = fir.load {{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
! HOST: fir.box_addr %[[DELETE_BOX]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_exit_data_region

! HOST-NOT: call @__fnacc_update_device(
! HOST-NOT: call @__fnacc_update_host(
! HOST-NOT: fnacc.create
! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.delete

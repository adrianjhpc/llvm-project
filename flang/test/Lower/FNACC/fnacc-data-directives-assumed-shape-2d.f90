! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_assumed_shape_2d(a)
  real :: a(:, :)

  !$fnacc update device(a)
  !$fnacc update host(a)
  !$fnacc release(a)
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_desc
! HOST-DAG: func.func private @__fnacc_update_host_desc
! HOST-DAG: func.func private @__fnacc_release_desc

! HOST-LABEL: func.func @_QPdata_assumed_shape_2d

! HOST: fir.box_addr
! HOST: fir.box_dims
! HOST: fir.box_dims
! HOST: call @__fnacc_update_device_desc

! HOST: fir.box_addr
! HOST: fir.box_dims
! HOST: fir.box_dims
! HOST: call @__fnacc_update_host_desc

! HOST: fir.box_addr
! HOST: call @__fnacc_release_desc

! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.release


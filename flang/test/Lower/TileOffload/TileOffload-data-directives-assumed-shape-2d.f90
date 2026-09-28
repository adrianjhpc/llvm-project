! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_assumed_shape_2d(a)
  real :: a(:, :)

  !$tileoff update device(a)
  !$tileoff update host(a)
  !$tileoff release(a)
end subroutine

! HOST-DAG: func.func private @__tileoff_update_device_desc
! HOST-DAG: func.func private @__tileoff_update_host_desc
! HOST-DAG: func.func private @__tileoff_release_desc

! HOST-LABEL: func.func @_QPdata_assumed_shape_2d

! HOST: fir.box_addr
! HOST: fir.box_dims
! HOST: fir.box_dims
! HOST: call @__tileoff_update_device_desc

! HOST: fir.box_addr
! HOST: fir.box_dims
! HOST: fir.box_dims
! HOST: call @__tileoff_update_host_desc

! HOST: fir.box_addr
! HOST: call @__tileoff_release_desc

! HOST-NOT: TileOffload.update_device
! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.release


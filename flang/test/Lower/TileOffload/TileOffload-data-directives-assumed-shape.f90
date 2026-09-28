! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_assumed_shape(a, c)
  real :: a(:, :)
  real :: c(:, :)

  !$tileoff update host(c)
  !$tileoff update device(a)
  !$tileoff release(a, c)
  !$tileoff release all
end subroutine

! HOST-DAG: func.func private @__tileoff_update_host_desc(!fir.ref<i8>, i64, i32, i64, i64, i64, i64, i64, i64)
! HOST-DAG: func.func private @__tileoff_update_device_desc(!fir.ref<i8>, i64, i32, i64, i64, i64, i64, i64, i64)
! HOST-DAG: func.func private @__tileoff_release_desc(!fir.ref<i8>)
! HOST-DAG: func.func private @__tileoff_release_all()

! HOST-LABEL: func.func @_QPdata_assumed_shape

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__tileoff_update_host_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__tileoff_update_device_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__tileoff_release_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__tileoff_release_desc(

! HOST: call @__tileoff_release_all

! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.update_device
! HOST-NOT: TileOffload.release
! HOST-NOT: TileOffload.release_all


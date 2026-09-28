! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_explicit_shape(n, a, c)
  integer :: n
  real :: a(n), c(n)

  !$tileoff update device(a)
  !$tileoff update host(c)
  !$tileoff release(a, c)
end subroutine

! HOST-DAG: func.func private @__tileoff_update_device_bytes
! HOST-DAG: func.func private @__tileoff_update_host_bytes
! HOST-DAG: func.func private @__tileoff_release

! HOST-LABEL: func.func @_QPdata_explicit_shape
! HOST: call @__tileoff_update_device_bytes
! HOST: call @__tileoff_update_host_bytes
! HOST: call @__tileoff_release
! HOST: call @__tileoff_release

! HOST-NOT: TileOffload.update_device
! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.release


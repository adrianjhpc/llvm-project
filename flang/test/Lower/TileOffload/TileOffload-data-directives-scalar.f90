! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_scalar(alpha)
  real :: alpha

  !$tileoff update device(alpha)
  !$tileoff update host(alpha)
  !$tileoff release(alpha)
end subroutine

! HOST-DAG: func.func private @__tileoff_update_device_bytes
! HOST-DAG: func.func private @__tileoff_update_host_bytes
! HOST-DAG: func.func private @__tileoff_release

! HOST-LABEL: func.func @_QPdata_scalar
! HOST: arith.constant 4 : i64
! HOST: call @__tileoff_update_device_bytes
! HOST: arith.constant 4 : i64
! HOST: call @__tileoff_update_host_bytes
! HOST: call @__tileoff_release

! HOST-NOT: TileOffload.update_device
! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.release


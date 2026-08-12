! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_scalar(alpha)
  real :: alpha

  !$fnacc update device(alpha)
  !$fnacc update host(alpha)
  !$fnacc release(alpha)
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_bytes
! HOST-DAG: func.func private @__fnacc_update_host_bytes
! HOST-DAG: func.func private @__fnacc_release

! HOST-LABEL: func.func @_QPdata_scalar
! HOST: arith.constant 4 : i64
! HOST: call @__fnacc_update_device_bytes
! HOST: arith.constant 4 : i64
! HOST: call @__fnacc_update_host_bytes
! HOST: call @__fnacc_release

! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.release


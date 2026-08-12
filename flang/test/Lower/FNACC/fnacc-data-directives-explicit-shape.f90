! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_explicit_shape(n, a, c)
  integer :: n
  real :: a(n), c(n)

  !$fnacc update device(a)
  !$fnacc update host(c)
  !$fnacc release(a, c)
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_bytes
! HOST-DAG: func.func private @__fnacc_update_host_bytes
! HOST-DAG: func.func private @__fnacc_release

! HOST-LABEL: func.func @_QPdata_explicit_shape
! HOST: call @__fnacc_update_device_bytes
! HOST: call @__fnacc_update_host_bytes
! HOST: call @__fnacc_release
! HOST: call @__fnacc_release

! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.release


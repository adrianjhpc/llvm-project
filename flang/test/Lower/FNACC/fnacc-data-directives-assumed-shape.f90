! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine data_assumed_shape(a, c)
  real :: a(:, :)
  real :: c(:, :)

  !$fnacc update host(c)
  !$fnacc update device(a)
  !$fnacc release(a, c)
  !$fnacc release all
end subroutine

! HOST-DAG: func.func private @__fnacc_update_host_desc(!fir.ref<i8>, i64, i32, i64, i64, i64, i64, i64, i64)
! HOST-DAG: func.func private @__fnacc_update_device_desc(!fir.ref<i8>, i64, i32, i64, i64, i64, i64, i64, i64)
! HOST-DAG: func.func private @__fnacc_release_desc(!fir.ref<i8>)
! HOST-DAG: func.func private @__fnacc_release_all()

! HOST-LABEL: func.func @_QPdata_assumed_shape

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__fnacc_update_host_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__fnacc_update_device_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__fnacc_release_desc(

! HOST: fir.box_addr
! HOST: fir.convert
! HOST: call @__fnacc_release_desc(

! HOST: call @__fnacc_release_all

! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.release
! HOST-NOT: fnacc.release_all


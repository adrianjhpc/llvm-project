! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine update_then_launch(a, b, c)
  real :: a(:), b(:), c(:)
  integer :: i

  !$fnacc update device(a)
  !$fnacc update device(b)

  !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
  do i = 1, size(c)
    c(i) = a(i) + b(i)
  end do

  !$fnacc update host(c)
  !$fnacc release all
end subroutine

! HOST-DAG: func.func private @__fnacc_update_device_desc
! HOST-DAG: func.func private @__fnacc_update_host_desc
! HOST-DAG: func.func private @__fnacc_launch_f32_v1
! HOST-DAG: func.func private @__fnacc_release_all

! HOST-LABEL: func.func @_QPupdate_then_launch
! HOST: call @__fnacc_update_device_desc
! HOST: call @__fnacc_update_device_desc
! HOST: call @__fnacc_launch_f32_v1
! HOST: call @__fnacc_update_host_desc
! HOST: call @__fnacc_release_all

! HOST-NOT: fnacc.update_device
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.launch
! HOST-NOT: fnacc.release_all


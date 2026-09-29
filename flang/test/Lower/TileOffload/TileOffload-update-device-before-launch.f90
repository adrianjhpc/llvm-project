! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --tileoff-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine update_then_launch(a, b, c)
  real :: a(:), b(:), c(:)
  integer :: i

  !$tileoff update device(a)
  !$tileoff update device(b)

  !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
  do i = 1, size(c)
    c(i) = a(i) + b(i)
  end do

  !$tileoff update host(c)
  !$tileoff release all
end subroutine

! HOST-DAG: func.func private @__tileoff_update_device_desc
! HOST-DAG: func.func private @__tileoff_update_host_desc
! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_release_all

! HOST-LABEL: func.func @_QPupdate_then_launch
! HOST: call @__tileoff_update_device_desc
! HOST: call @__tileoff_update_device_desc
! HOST: call @__tileoff_begin_launch_v2
! HOST: call @__tileoff_update_host_desc
! HOST: call @__tileoff_release_all

! HOST-NOT: TileOffload.update_device
! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.launch
! HOST-NOT: TileOffload.release_all

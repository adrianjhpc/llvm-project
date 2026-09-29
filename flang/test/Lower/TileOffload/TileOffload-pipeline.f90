! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --tileoff-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json emit-fortran-aliases=true" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine tileoff_pipeline_test(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$tileoff parallel tile(128) no_copyback pack(a:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$tileoff update host(c)
  !$tileoff release all
end subroutine

! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_bind_array_v2
! HOST-DAG: func.func private @__tileoff_commit_launch_v2
! HOST-DAG: func.func private @__tileoff_update_host
! HOST-DAG: func.func private @__tileoff_release_all


! HOST-LABEL: func.func @_QPtileoff_pipeline_test
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-3: call @__tileoff_bind_array_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST: call @__tileoff_update_host
! HOST: call @__tileoff_release_all

! HOST: func.func @tileoff_pipeline_test_

! HOST-NOT: TileOffload.launch
! HOST-NOT: TileOffload.update_host
! HOST-NOT: TileOffload.release_all

! TTIR: module attributes
! TTIR: tt.func @tileoff_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "tileoff_kernel_0"
! JSON: "kind": "binary"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]
! JSON: "launch_abi_version": 2
! JSON: "array_count": 3
! JSON: "scalar_count": 0
! JSON: "output_count": 1
! JSON: "copy_back_writes": false
! JSON: "pack": [
! JSON: "kernel_arg_slot": 0
! JSON: "target": 1
! JSON: "target_name": "device"

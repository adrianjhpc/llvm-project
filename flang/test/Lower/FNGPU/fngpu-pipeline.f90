! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fngpu-pipeline="ttir-output=%t.ttir json-output=%t.json emit-fortran-aliases=true" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fngpu_pipeline_test(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fngpu parallel tile(128) pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$fngpu update host(c)
  !$fngpu release all
end subroutine

! HOST-DAG: func.func private @__fngpu_launch_f32_v1
! HOST-DAG: func.func private @__fngpu_update_host
! HOST-DAG: func.func private @__fngpu_release_all


! HOST-LABEL: func.func @_QPfngpu_pipeline_test
! HOST: call @__fngpu_launch_f32_v1
! HOST: call @__fngpu_update_host
! HOST: call @__fngpu_release_all

! HOST: func.func @fngpu_pipeline_test_

! HOST-NOT: fngpu.launch
! HOST-NOT: fngpu.update_host
! HOST-NOT: fngpu.release_all

! TTIR: module attributes
! TTIR: tt.func @fngpu_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "fngpu_kernel_0"
! JSON: "kind": "binary"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]
! JSON: "pack": [
! JSON: "kernel_arg_slot": 0
! JSON: "target": 1
! JSON: "target_name": "device"
! JSON: "kernel_arg_slot": 2
! JSON: "target": 1
! JSON: "target_name": "device"


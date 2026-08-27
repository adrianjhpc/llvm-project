! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json emit-fortran-aliases=true" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_pipeline_test(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128) pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$fnacc update host(c)
  !$fnacc release all
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-DAG: func.func private @__fnacc_update_host
! HOST-DAG: func.func private @__fnacc_release_all


! HOST-LABEL: func.func @_QPfnacc_pipeline_test
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-3: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST: call @__fnacc_update_host
! HOST: call @__fnacc_release_all

! HOST: func.func @fnacc_pipeline_test_

! HOST-NOT: fnacc.launch
! HOST-NOT: fnacc.update_host
! HOST-NOT: fnacc.release_all

! TTIR: module attributes
! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "kind": "binary"
! JSON: "launch_abi_version": 2
! JSON: "array_count": 3
! JSON: "scalar_count": 0
! JSON: "output_count": 1
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]
! JSON: "pack": [
! JSON: "kernel_arg_slot": 0
! JSON: "target": 1
! JSON: "target_name": "device"
! JSON: "kernel_arg_slot": 2
! JSON: "target": 1
! JSON: "target_name": "device"

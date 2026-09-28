! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine tileoff_sum_reduce(n, a, sum)
  integer :: n
  real :: a(n)
  real :: sum
  integer :: i

  sum = 0.0

  !$tileoff parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i)
  end do
end subroutine

! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_bind_array_v2
! HOST-DAG: func.func private @__tileoff_bind_reduction_result_f32_v2
! HOST-DAG: func.func private @__tileoff_commit_launch_v2

! HOST-LABEL: func.func @_QPtileoff_sum_reduce
! HOST: call @__tileoff_begin_launch_v2
! HOST: call @__tileoff_bind_array_v2
! HOST: call @__tileoff_bind_reduction_result_f32_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST-NOT: TileOffload.launch

! TTIR-LABEL: tt.func @tileoff_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %partials: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: tt.load
! TTIR: tt.reduce
! TTIR: tt.store

! TTIR-LABEL: tt.func @tileoff_kernel_0_reduce_stage
! TTIR-SAME: %input: !tt.ptr<f32>
! TTIR-SAME: %output: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: tt.load
! TTIR: tt.reduce
! TTIR: tt.store

! JSON: "tileoff_schema_version": 1
! JSON-DAG: "kind": "reduction_sum1d"
! JSON-DAG: "launch_abi_version": 2
! JSON-DAG: "array_count": 1
! JSON-DAG: "output_count": 1
! JSON-DAG: "reduction_op": "add"
! JSON-DAG: "reduction_stage_id": 1
! JSON-DAG: "rank": 1
! JSON-DAG: "tile": [256, 1, 1]
! JSON-DAG: "role": "partials"
! JSON-DAG: "name": "tileoff_kernel_0_reduce_stage"
! JSON-DAG: "kind": "reduction_stage1d"

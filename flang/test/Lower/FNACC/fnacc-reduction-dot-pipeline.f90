! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_dot_reduce(n, a, b, sum)
  integer :: n
  real :: a(n), b(n)
  real :: sum
  integer :: i

  sum = 0.0

  !$fnacc parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i) * b(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_launch_reduce_f32_v2

! HOST-LABEL: func.func @_QPfnacc_dot_reduce
! HOST: call @__fnacc_launch_reduce_f32_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %partials: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: tt.load
! TTIR: tt.load
! TTIR: arith.mulf
! TTIR: tt.reduce
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_0_reduce_stage
! TTIR-SAME: %input: !tt.ptr<f32>
! TTIR-SAME: %output: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: tt.load
! TTIR: tt.reduce
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "kind": "reduction_dot1d"
! JSON-DAG: "reduction_stage_id": 1
! JSON-DAG: "rank": 1
! JSON-DAG: "tile": [256, 1, 1]
! JSON-DAG: "role": "partials"
! JSON-DAG: "name": "fnacc_kernel_0_reduce_stage"
! JSON-DAG: "kind": "reduction_stage1d"

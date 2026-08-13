! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_sum_reduce(n, a, sum)
  integer :: n
  real :: a(n)
  real :: sum
  integer :: i

  sum = 0.0

  !$fnacc parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_launch_reduce_f32_v1

! HOST-LABEL: func.func @_QPfnacc_sum_reduce
! HOST: call @__fnacc_launch_reduce_f32_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %partials: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: tt.load
! TTIR: tt.reduce
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "kind": "reduction_sum1d"
! JSON-DAG: "rank": 1
! JSON-DAG: "tile": [256, 1, 1]
! JSON-DAG: "role": "partials"


! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_reduction_expr_variadic(n, a, b, c, d, e, &
                                         alpha, beta, gamma, delta, result)
  integer :: n
  real :: a(n), b(n), c(n), d(n), e(n)
  real :: alpha, beta, gamma, delta, result
  integer :: i

  result = 0.0
  !$fnacc parallel tile(256) reduction(+:result)
  do i = 1, n
    result = result + (alpha * a(i) + beta * b(i) + gamma * c(i) + &
                       delta * d(i) + e(i))
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f32_v2
! HOST-DAG: func.func private @__fnacc_bind_reduction_result_f32_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2

! HOST-LABEL: func.func @_QPfnacc_reduction_expr_variadic
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-5: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_bind_reduction_result_f32_v2
! HOST-COUNT-4: call @__fnacc_bind_scalar_f32_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %read1: !tt.ptr<f32>
! TTIR-SAME: %read2: !tt.ptr<f32>
! TTIR-SAME: %read3: !tt.ptr<f32>
! TTIR-SAME: %read4: !tt.ptr<f32>
! TTIR-SAME: %partials: !tt.ptr<f32>
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR-SAME: %scalar2: f32
! TTIR-SAME: %scalar3: f32
! TTIR-SAME: %n: i32
! TTIR-COUNT-5: tt.load
! TTIR-COUNT-4: tt.splat %scalar
! TTIR: tt.reduce
! TTIR: tt.store

! JSON: "kind": "reduction_sum1d"
! JSON: "launch_abi_version": 2
! JSON: "array_count": 5
! JSON: "scalar_count": 4
! JSON: "output_count": 1
! JSON: "role": "partials"
! JSON-COUNT-4: "role": "scalar"

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr_mixed_types(n, a, b, c)
  integer :: n
  real :: a(n), c(n)
  real(8) :: b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + real(b(i))
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-COUNT-1: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f32>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %array2: !tt.ptr<f32>
! TTIR: tt.load {{.*}} : tensor<128x!tt.ptr<f32>>
! TTIR: tt.load {{.*}} : tensor<128x!tt.ptr<f64>>
! TTIR: arith.truncf {{.*}} : tensor<128xf64> to tensor<128xf32>
! TTIR: arith.addf {{.*}} : tensor<128xf32>
! TTIR: tt.store {{.*}} : tensor<128x!tt.ptr<f32>>

! JSON: "fnacc_schema_version": 1
! JSON: "kind": "multi_expr1d"
! JSON: "array_count": 3
! JSON-DAG: "type": "ptr<f32>"
! JSON-DAG: "type": "ptr<f64>"
! JSON-DAG: "type": "ptr<f32>"
! JSON-DAG: "type": "i32"

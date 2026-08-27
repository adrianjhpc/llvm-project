! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine variadic_expr(n, a, b, c, d, out, s0, s1, s2, s3)
  integer :: n
  real :: a(n), b(n), c(n), d(n), out(n)
  real :: s0, s1, s2, s3
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    out(i) = a(i) * s0 + b(i) * s1 + c(i) * s2 + d(i) * s3
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f32_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPvariadic_expr
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-5: call @__fnacc_bind_array_v2
! HOST-COUNT-4: call @__fnacc_bind_scalar_f32_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %read1: !tt.ptr<f32>
! TTIR-SAME: %read2: !tt.ptr<f32>
! TTIR-SAME: %read3: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR-SAME: %scalar2: f32
! TTIR-SAME: %scalar3: f32
! TTIR-SAME: %n: i32
! TTIR: tt.splat %read3
! TTIR: tt.splat %scalar3
! TTIR: tt.store

! JSON: "kind": "expr1d"
! JSON: "launch_abi_version": 2
! JSON: "array_count": 5
! JSON: "scalar_count": 4
! JSON: "output_count": 1
! JSON: "name": "read3"
! JSON-SAME: "array_index": 3
! JSON: "name": "scalar3"
! JSON-SAME: "scalar_index": 3

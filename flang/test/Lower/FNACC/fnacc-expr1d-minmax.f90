! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr1d_min(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = min(a(i), b(i))
  end do
end subroutine

subroutine fnacc_expr1d_max(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = max(a(i), b(i))
  end do
end subroutine

subroutine fnacc_expr1d_clamp(n, a, lower, upper, c)
  integer :: n
  real :: a(n), c(n)
  real :: lower, upper
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = min(max(a(i), lower), upper)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-COUNT-3: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! Flang currently lowers the MIN/MAX intrinsics to comparisons and selects.
! Keeping that form preserves its exact NaN and signed-zero behavior.  Direct
! arith.minimumf/maximumf inputs are also supported by FNACC, but these
! source-level tests must check the FIR form that Flang actually produces.

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: arith.cmpf
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: arith.cmpf
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR: arith.cmpf
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: arith.cmpf
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-COUNT-3: "kind": "expr1d"
! JSON: "name": "scalar0"
! JSON: "type": "f32"
! JSON: "name": "scalar1"
! JSON: "type": "f32"

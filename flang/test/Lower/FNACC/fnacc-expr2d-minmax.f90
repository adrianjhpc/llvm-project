! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr2d_minmax(n, m, a, b, c)
  integer :: n, m
  real :: a(n, m), b(n, m), c(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      c(i, j) = min(a(i, j), b(i, j))
    end do
  end do
end subroutine

subroutine fnacc_expr2d_clamp(n, m, a, lower, upper, c)
  integer :: n, m
  real :: a(n, m), c(n, m)
  real :: lower, upper
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      c(i, j) = max(lower, min(a(i, j), upper))
    end do
  end do
end subroutine

! HOST-COUNT-2: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! Flang lowers source-level MIN/MAX intrinsics to comparison/select trees so
! that their NaN and signed-zero behavior remains explicit in FIR and TTIR.

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: arith.cmpf
! TTIR-SAME: tensor<256xf32>
! TTIR: arith.select
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR: arith.cmpf
! TTIR-SAME: tensor<256xf32>
! TTIR: arith.select
! TTIR: arith.cmpf
! TTIR-SAME: tensor<256xf32>
! TTIR: arith.select
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-COUNT-2: "kind": "expr2d"
! JSON: "rank": 2
! JSON: "role": "scalar"
! JSON: "name": "scalar0"
! JSON: "role": "scalar"
! JSON: "name": "scalar1"

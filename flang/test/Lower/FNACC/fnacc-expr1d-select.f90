! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr1d_select_zero(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = merge(a(i), b(i), a(i) > 0.0)
  end do
end subroutine

subroutine fnacc_expr1d_select_arrays(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = merge(a(i), b(i), a(i) >= b(i))
  end do
end subroutine

! HOST-COUNT-2: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: arith.constant 0.000000e+00 : f32
! TTIR: arith.cmpf ogt
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: arith.cmpf oge
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.select
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-COUNT-2: "kind": "expr1d"
! JSON: "rank": 1

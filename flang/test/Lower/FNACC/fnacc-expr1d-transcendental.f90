! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr1d_exp(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = exp(a(i))
  end do
end subroutine

subroutine fnacc_expr1d_log(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = log(a(i))
  end do
end subroutine

subroutine fnacc_expr1d_sin_cos(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = sin(a(i)) + cos(a(i))
  end do
end subroutine

subroutine fnacc_expr1d_tanh(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = tanh(a(i))
  end do
end subroutine

! HOST-COUNT-4: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: math.exp
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: math.log
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR: math.sin
! TTIR-SAME: tensor<128xf32>
! TTIR: math.cos
! TTIR-SAME: tensor<128xf32>
! TTIR: arith.addf
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_3(
! TTIR: math.tanh
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-COUNT-4: "kind": "expr1d"
! JSON-COUNT-4: "rank": 1
! JSON-COUNT-4: "tile": [128, 1, 1]


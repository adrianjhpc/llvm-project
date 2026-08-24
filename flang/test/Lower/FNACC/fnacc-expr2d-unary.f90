! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr2d_unary(n, m, a, b)
  integer :: n, m
  real :: a(n, m), b(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      b(i, j) = sqrt(abs(a(i, j)))
    end do
  end do
end subroutine

subroutine fnacc_expr2d_neg_exp(n, m, a, b)
  integer :: n, m
  real :: a(n, m), b(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      b(i, j) = -exp(a(i, j))
    end do
  end do
end subroutine

! HOST-COUNT-2: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: tt.get_program_id x
! TTIR: tt.get_program_id y
! TTIR: math.absf
! TTIR-SAME: tensor<256xf32>
! TTIR: math.sqrt
! TTIR-SAME: tensor<256xf32>
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: math.exp
! TTIR-SAME: tensor<256xf32>
! TTIR: arith.negf
! TTIR-SAME: tensor<256xf32>
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON-COUNT-2: "kind": "expr2d"
! JSON-COUNT-2: "rank": 2
! JSON-COUNT-2: "tile": [16, 16, 1]


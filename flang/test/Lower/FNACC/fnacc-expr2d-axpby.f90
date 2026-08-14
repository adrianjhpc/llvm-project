! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine expr2d_axpby(n, m, alpha, beta, a, b, c)
  integer :: n, m
  real :: alpha, beta
  real :: a(n, m), b(n, m), c(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      c(i, j) = alpha * a(i, j) + beta * b(i, j)
    end do
  end do
end subroutine

! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %read1: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: arith.mulf
! TTIR: math.fma
! TTIR: tt.store

! JSON: "kind": "expr2d"
! JSON: "rank": 2


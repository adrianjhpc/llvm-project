! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr2d_select(n, m, a, b, c)
  integer :: n, m
  real :: a(n, m), b(n, m), c(n, m)
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      c(i, j) = merge(a(i, j), b(i, j), a(i, j) >= b(i, j))
    end do
  end do
end subroutine

! HOST: call @__fnacc_launch_f32_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %read1: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: arith.cmpf oge
! TTIR-SAME: tensor<256xf32>
! TTIR: arith.select
! TTIR: tt.store

! JSON: "fnacc_schema_version": 1
! JSON: "kind": "expr2d"
! JSON: "rank": 2
! JSON: "tile": [16, 16, 1]


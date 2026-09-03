! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json

! Verify that an SSA value used by several parts of a stencil expression is
! emitted once.  FIR is a DAG, but the recognizer represents expressions as
! owning trees; losing the shared identity here used to duplicate `a+b` once
! for every use below.
subroutine stencil_expression_dag(x_min, x_max, y_min, y_max, a, b, c)
  implicit none
  integer :: x_min, x_max, y_min, y_max, j, k
  real(8) :: a(x_min-1:x_max+1, y_min-1:y_max+1)
  real(8) :: b(x_min-1:x_max+1, y_min-1:y_max+1)
  real(8) :: c(x_min-1:x_max+1, y_min-1:y_max+1)
  real(8) :: tmp

  !$fnacc parallel tile(16,16)
  do k = y_min, y_max
    do j = x_min, x_max
      tmp = a(j,k) + b(j,k)
      c(j,k) = merge(tmp, b(j,k), tmp < b(j,k)) + tmp
    end do
  end do
end subroutine

! TTIR: %[[TMP:expr[0-9]+]] = arith.addf
! TTIR: %[[PRED:pred[0-9]+]] = arith.cmpf olt, %[[TMP]],
! TTIR: %[[SELECTED:expr[0-9]+]] = arith.select %[[PRED]], %[[TMP]],
! TTIR: arith.addf %[[SELECTED]], %[[TMP]]
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "array_count": 3

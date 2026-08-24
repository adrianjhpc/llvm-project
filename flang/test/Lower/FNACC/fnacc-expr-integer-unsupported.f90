! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o /dev/null 2>&1 | FileCheck %s

subroutine fnacc_expr_integer_unsupported(n, a, b, c)
  integer :: n
  integer :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine

! CHECK:  FNACC Triton cannot emit launch: not a supported FNACC kernel; matmul failure: matmul found no i loop; 2-D failure: 2-D recognition found no inner loop; 1-D failure: unsupported 1-D elementwise expression; binary failure: stored value is not a supported binary floating-point op; SAXPY failure: SAXPY pattern requires stored value to be arith.addf; expression-tree failure: elementwise expression has unsupported result type


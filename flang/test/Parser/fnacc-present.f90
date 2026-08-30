! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

subroutine present_parser_test(a, b)
  real :: a(:), b(:)
  !$fnacc present(a, b)
end subroutine

! CHECK: !$FNACC PRESENT(a, b)

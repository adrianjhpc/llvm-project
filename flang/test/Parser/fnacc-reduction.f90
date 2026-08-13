! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fnacc_reduction(n, a, b, sum)
  integer :: n
  real :: a(n), b(n)
  real :: sum
  integer :: i

  !$fnacc parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i) * b(i)
  end do
end subroutine

! CHECK: FnACCConstruct
! CHECK: FnACCParallelDirective
! CHECK: FnACCTileClause
! CHECK: FnACCReductionClause
! CHECK: FnACCReductionOperator
! CHECK: Name = 'sum'


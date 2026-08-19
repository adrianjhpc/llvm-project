! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=UNPARSE

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

subroutine test_fnacc_reduction_operators(n, a, product, smallest, largest)
  integer :: n
  real :: a(n), product, smallest, largest
  integer :: i

  !$fnacc parallel tile(256) reduction(*:product)
  do i = 1, n
    product = product * a(i)
  end do

  !$fnacc parallel tile(256) reduction(min:smallest)
  do i = 1, n
    smallest = min(smallest, a(i))
  end do

  !$fnacc parallel tile(256) reduction(max:largest)
  do i = 1, n
    largest = max(largest, a(i))
  end do
end subroutine

! CHECK: FnACCConstruct
! CHECK: FnACCParallelDirective
! CHECK: FnACCTileClause
! CHECK: FnACCReductionClause
! CHECK: FnACCReductionOperator
! CHECK: Name = 'sum'
! CHECK: Name = 'product'
! CHECK: Name = 'smallest'
! CHECK: Name = 'largest'

! UNPARSE: !$FNACC PARALLEL TILE(256_4) REDUCTION(+:sum)
! UNPARSE: !$FNACC PARALLEL TILE(256_4) REDUCTION(*:product
! UNPARSE: !$FNACC PARALLEL TILE(256_4) REDUCTION(MIN:smallest)
! UNPARSE: !$FNACC PARALLEL TILE(256_4) REDUCTION(MAX:largest)


! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fnacc_data_directives(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128) no_copyback pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$fnacc update host(c)
  !$fnacc update device(a)
  !$fnacc release(a, c)
  !$fnacc release all
end subroutine

! CHECK: FnACCConstruct
! CHECK: FnACCParallelDirective
! CHECK: FnACCTileClause
! CHECK: FnACCNoCopybackClause
! CHECK: FnACCPackClause

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCUpdateHostDirective
! CHECK: Name = 'c'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCUpdateDeviceDirective
! CHECK: Name = 'a'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCReleaseDirective
! CHECK: Name = 'a'
! CHECK: Name = 'c'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCReleaseAllDirective


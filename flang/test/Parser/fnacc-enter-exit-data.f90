! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fnacc_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCEnterDataDirective
! CHECK: FnACCCopyinClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: FnACCCreateClause
! CHECK: Name = 'c'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCExitDataDirective
! CHECK: FnACCCopyoutClause
! CHECK: Name = 'c'
! CHECK: FnACCDeleteClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: Name = 'c'


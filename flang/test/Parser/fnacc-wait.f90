! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fnacc_wait()
  !$fnacc wait
end subroutine

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCWaitDirective

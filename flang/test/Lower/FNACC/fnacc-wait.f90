! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine fnacc_wait_test()
  !$fnacc wait
end subroutine fnacc_wait_test

! CHECK-LABEL: func.func @_QPfnacc_wait_test
! CHECK: fnacc.wait
! CHECK: return

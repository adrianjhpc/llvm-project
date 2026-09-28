! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine tileoff_wait_test()
  !$tileoff wait
end subroutine tileoff_wait_test

! CHECK-LABEL: func.func @_QPtileoff_wait_test
! CHECK: TileOffload.wait
! CHECK: return

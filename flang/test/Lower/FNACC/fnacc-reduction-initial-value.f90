! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-assign-kernel-ids \
! RUN:   --fnacc-lower-to-runtime %t.fir | FileCheck %s

subroutine reduce_with_initial(n, a, total)
  integer :: n, i
  real :: a(n), total

  total = 5.0

  !$fnacc parallel tile(256) reduction(+:total)
  do i = 1, n
    total = total + a(i)
  end do
end subroutine

! CHECK: %[[INITIAL:[0-9]+]] = fir.load %[[RESULT:[0-9]+]] : !fir.ref<f32>
! CHECK: call @__fnacc_launch_reduce_f32_v2
! CHECK-SAME: %[[RESULT]], %[[INITIAL]],


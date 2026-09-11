! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir 2>&1 | FileCheck %s
subroutine matmul_step(a,b,c,step)
  real :: a(32,32),b(32,32),c(32,32),acc
  integer :: i,j,p,step
  !$fnacc parallel tile(16,16,8)
  do j=1,32
    do i=1,32
      acc=0.0
      do p=1,32,step
        acc=acc+a(i,p)*b(p,j)
      enddo
      c(i,j)=acc
    enddo
  enddo
end subroutine
! CHECK: matmul p loop step must be a nonzero constant signed 32-bit integer

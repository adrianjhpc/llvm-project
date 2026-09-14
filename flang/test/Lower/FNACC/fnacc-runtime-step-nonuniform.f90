! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s
subroutine nonuniform_step(a,b,n)
 integer :: n,i,j
 real :: a(n,n),b(n,n)
 !$fnacc parallel tile(16,4)
 do j=1,n
  do i=1,n,j
   b(i,j)=a(i,j)*2.0
  enddo
 enddo
end subroutine
! CHECK: runtime loop step must be invariant across the launch

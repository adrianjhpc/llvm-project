! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.2.ttir json-output=%t.2.json" %t.fir -o %t.2.host
! RUN: FileCheck %s --check-prefix=TTIR < %t.2.ttir
! RUN: FileCheck %s --check-prefix=JSON < %t.2.json
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 ttir-output=%t.3.ttir json-output=%t.3.json" %t.fir -o %t.3.host
! RUN: FileCheck %s --check-prefix=TTIR < %t.3.ttir
! RUN: FileCheck %s --check-prefix=JSON < %t.3.json

subroutine step_vector(n,a,b)
 integer :: n,i
 real(8) :: a(1:n),b(1:n)
 !$fnacc parallel tile(16)
 do i=n,1,-2
  b(i)=a(i)*2.0_8
 enddo
end subroutine

subroutine step_reduction(n,m,a,s)
 integer :: n,m,i,j
 real(8) :: a(n,m),s
 !$fnacc parallel tile(16,4) reduction(+:s)
 do j=m,1,-3
  do i=1,n,2
   s=s+a(i,j)
  enddo
 enddo
end subroutine

subroutine step_matmul(n,a,b,c)
 integer :: n,i,j,p
 real :: a(n,n),b(n,n),c(n,n),acc
 !$fnacc parallel tile(16,16,8)
 do j=n,1,-3
  do i=1,n,2
   acc=0.0
   do p=n,1,-2
    acc=acc+a(i,p)*b(p,j)
   enddo
   c(i,j)=acc
  enddo
 enddo
end subroutine

! TTIR: %step_x_s = arith.constant dense<-2>
! TTIR: %scaled_x = arith.muli %offs, %step_x_s
! TTIR: %source_x = arith.addi %scaled_x, %loop_lower_x_s
! TTIR: %step_x_s = arith.constant dense<2>
! TTIR: %step_y_s = arith.constant dense<-3>
! TTIR: %a_addr_step_r = arith.constant dense<2>
! TTIR: %a_addr_step_c = arith.constant dense<-2>
! TTIR: %b_addr_step_r = arith.constant dense<-2>
! TTIR: %b_addr_step_c = arith.constant dense<-3>
! TTIR: %c_addr_step_r = arith.constant dense<2>
! TTIR: %c_addr_step_c = arith.constant dense<-3>
! JSON: "loop_steps": [-2, 1, 1]
! JSON: "loop_steps": [2, -3, 1]
! JSON: "loop_steps": [2, -3, -2]

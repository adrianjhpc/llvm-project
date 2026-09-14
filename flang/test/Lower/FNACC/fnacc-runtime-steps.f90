! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.2.ttir json-output=%t.2.json" %t.fir -o %t.2.host
! RUN: FileCheck %s --check-prefixes=HOST,V2 < %t.2.host
! RUN: FileCheck %s --check-prefix=DEVICE < %t.2.ttir
! RUN: FileCheck %s --check-prefix=JSON < %t.2.json
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 ttir-output=%t.3.ttir json-output=%t.3.json" %t.fir -o %t.3.host
! RUN: FileCheck %s --check-prefixes=HOST,V3 < %t.3.host
! RUN: FileCheck %s --check-prefix=DEVICE < %t.3.ttir
! RUN: FileCheck %s --check-prefix=JSON < %t.3.json
subroutine runtime_vector(a,b,lo,hi,step,alpha)
 real(8) :: a(-8:40),b(-8:40),alpha
 integer :: lo,hi,step,i
 !$fnacc parallel tile(16)
 do i=lo,hi,step
  b(i)=alpha*a(i)
 enddo
end subroutine
subroutine runtime_matmul(a,b,c,lo,hi,step)
 real :: a(-8:40,-8:40),b(-8:40,-8:40),c(-8:40,-8:40),acc
 integer :: lo(3),hi(3),step(3),i,j,p
 !$fnacc parallel tile(16,16,8)
 do j=lo(2),hi(2),step(2)
  do i=lo(1),hi(1),2
   acc=0
   do p=lo(3),hi(3),step(3)
    acc=acc+a(i,p)*b(p,j)
   enddo
   c(i,j)=acc
  enddo
 enddo
end subroutine
! HOST-LABEL: func.func @_QPruntime_vector
! HOST: call @__fnacc_trip_count_i32
! V2: call @__fnacc_bind_scalar_f64_v2
! V2: call @__fnacc_bind_scalar_i32_v2
! V2: call @__fnacc_commit_launch_v2
! V3: call @__fnacc_launch_v3
! HOST-LABEL: func.func @_QPruntime_matmul
! HOST-COUNT-2: call @__fnacc_trip_count_i32
! V2-COUNT-2: call @__fnacc_bind_scalar_i32_v2
! V2: call @__fnacc_commit_launch_v2
! V3: call @__fnacc_launch_v3
! DEVICE: %loop_step_0: i32) attributes
! DEVICE: %step_x_s = tt.splat %loop_step_0
! DEVICE: %loop_step_1: i32, %loop_step_2: i32) attributes
! DEVICE: %a_addr_step_r = arith.constant dense<2>
! DEVICE: %a_addr_step_c64 = arith.extsi %loop_step_2 : i32 to i64
! DEVICE: %b_addr_step_c64 = arith.extsi %loop_step_1 : i32 to i64
! JSON: "loop_steps": [0, 1, 1]
! JSON: "loop_step_scalar_indices": [1, -1, -1]
! JSON: "scalar_count": 2
! JSON: "loop_steps": [2, 0, 0]
! JSON: "loop_step_scalar_indices": [-1, 0, 1]
! JSON: "scalar_count": 2

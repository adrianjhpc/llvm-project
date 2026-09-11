! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host
! RUN: FileCheck %s --check-prefix=HOST < %t.host

subroutine partial_1d(a,b,lo,hi)
 real(8) :: a(-8:40),b(-8:40)
 integer :: lo,hi,i
 !$fnacc parallel tile(16)
 do i=lo,hi
  b(i)=2.0_8*a(i)
 enddo
end subroutine

subroutine partial_2d(a,b,lx,ux,ly,uy)
 real(8) :: a(-8:40,-8:40),b(-8:40,-8:40)
 integer :: lx,ux,ly,uy,i,j
 !$fnacc parallel tile(16,4)
 do j=ly,uy
  do i=lx,ux
   b(i,j)=2.0_8*a(i,j)
  enddo
 enddo
end subroutine

! The output binding must carry read/write flags (3), even with unit steps.
! HOST-LABEL: func.func @_QPpartial_1d
! HOST: %[[RW1:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
! HOST: call @__fnacc_bind_array_v2({{.*}}, %[[RW1]],
! HOST-LABEL: func.func @_QPpartial_2d
! HOST: %[[RW2:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
! HOST: call @__fnacc_bind_array_v2({{.*}}, %[[RW2]],

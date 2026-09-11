! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=%t.2.ttir json-output=%t.2.json" %t.fir -o %t.2.host
! RUN: FileCheck %s --check-prefix=HOST --implicit-check-not=fnacc.launch < %t.2.host
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 ttir-output=%t.3.ttir json-output=%t.3.json" %t.fir -o %t.3.host
! RUN: FileCheck %s --check-prefix=HOST --implicit-check-not=fnacc.launch < %t.3.host

subroutine array_bounds_1d(a,b,lo,hi)
 real(8) :: a(-8:40),b(-8:40)
 integer :: lo(2),hi(2),i
 !$fnacc parallel tile(16)
 do i=lo(1),hi(1),2
   b(i)=a(i)*2.0_8
 enddo
 ! Also exercise the LoadIntegerRef nested inside bound arithmetic.
 !$fnacc parallel tile(16)
 do i=hi(1)-1,lo(1)+1,-2
   b(i)=a(i)*3.0_8
 enddo
end subroutine

subroutine array_bounds_2d(a,b,lo,hi)
 real(8) :: a(-8:40,-8:40),b(-8:40,-8:40)
 integer :: lo(2),hi(2),i,j
 !$fnacc parallel tile(16,4)
 do j=hi(2),lo(2),-3
  do i=lo(1),hi(1),2
   b(i,j)=a(i,j)*2.0_8
  enddo
 enddo
end subroutine

! HOST-LABEL: func.func @_QParray_bounds_1d
! HOST: fir.array_coor
! HOST: fir.load
! HOST: call @__fnacc_{{begin_launch_v2|launch_v3}}
! HOST: call @__fnacc_{{begin_launch_v2|launch_v3}}
! HOST-LABEL: func.func @_QParray_bounds_2d
! HOST: fir.array_coor
! HOST: fir.load
! HOST: call @__fnacc_{{begin_launch_v2|launch_v3}}

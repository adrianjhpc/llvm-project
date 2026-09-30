! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --tileoff-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json f64-matmul-strategy=dot" %t.fir -o /dev/null
! RUN: FileCheck %s --check-prefix=IR < %t.ttir
subroutine matmul_unit_stride(a,b,c)
  real(8) :: a(:,:),b(:,:),c(:,:),acc
  integer :: i,j,k
  !$tileoff parallel tile(32,16)
  do j=1,size(c,2)
    do i=1,size(c,1)
      acc=0.0_8
      do k=1,size(a,2)
        acc=acc+a(i,k)*b(k,j)
      enddo
      c(i,j)=acc
    enddo
  enddo
end subroutine
! IR: tt.func @
! IR: %a_unit_row = arith.cmpi eq, %a_s0, %layout_one : i32
! IR: %b_unit_row = arith.cmpi eq, %b_s0, %layout_one : i32
! IR: %c_unit_row = arith.cmpi eq, %c_s0, %layout_one : i32
! IR: scf.if %abc_unit_row {
! IR: %a_addr_sr = arith.constant dense<1>
! IR: %b_addr_sr = arith.constant dense<1>
! IR: tt.dot
! IR: %c_addr_sr = arith.constant dense<1>
! IR: tt.store
! IR: } else {
! IR: %a_addr_sr64 = arith.extsi %a_s0 : i32 to i64
! IR: %b_addr_sr64 = arith.extsi %b_s0 : i32 to i64
! IR: tt.dot
! IR: %c_addr_sr64 = arith.extsi %c_s0 : i32 to i64
! IR: tt.store
! IR: tt.return

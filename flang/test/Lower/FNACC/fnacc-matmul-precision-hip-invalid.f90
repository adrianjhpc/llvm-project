! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json accelerator-target=hip" %t.fir -o /dev/null 2>&1 | FileCheck %s
! CHECK: TF32 matmul precision is currently supported only by the CUDA Triton backend

subroutine hip_precision(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) matmul_precision(tf32x3)
  do j = 1,size(c,2)
    do i = 1,size(c,1)
      acc = 0
      do p = 1,size(a,2)
        acc = acc + a(i,p)*b(p,j)
      end do
      c(i,j) = acc
    end do
  end do
end subroutine

! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: error:

subroutine invalid_mode(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) matmul_precision(tf32x2)
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

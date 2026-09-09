! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine duplicate(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !ERROR: FNACC MATMUL_PRECISION clause may appear at most once
  !$fnacc parallel tile(64,64,32) matmul_precision(ieee) matmul_precision(tf32)
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

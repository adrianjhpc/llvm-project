! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=AST
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=UNPARSE

subroutine parse_ieee(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) MaTmUl_PrEcIsIoN(IEEE)
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

! AST: FnACCMatmulPrecisionClause
! AST: FnACCMatmulPrecision
! UNPARSE: MATMUL_PRECISION(IEEE)

subroutine parse_tf32(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) MaTmUl_PrEcIsIoN(TF32)
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

! AST: FnACCMatmulPrecisionClause
! AST: FnACCMatmulPrecision
! UNPARSE: MATMUL_PRECISION(TF32)

subroutine parse_tf32x3(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) MaTmUl_PrEcIsIoN(TF32X3)
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

! AST: FnACCMatmulPrecisionClause
! AST: FnACCMatmulPrecision
! UNPARSE: MATMUL_PRECISION(TF32X3)

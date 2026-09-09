! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: not fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s
! CHECK: MATMUL_PRECISION requires a recognized real(4) matmul

subroutine wrong_kind(n,a,b,c)
  integer :: n,i
  real :: a(n),b(n),c(n)
  !$fnacc parallel tile(128) matmul_precision(ieee)
  do i = 1,n
    c(i) = a(i)+b(i)
  end do
end subroutine

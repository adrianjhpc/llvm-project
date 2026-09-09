! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: FileCheck %s --check-prefix=FIR --input-file=%t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine precision_default(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32)
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

! FIR-LABEL: func.func @_QPprecision_default
! FIR-NOT: fnacc.matmul_precision
! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: tt.load %a_ptrs, %mask_a, %a_zero
! TTIR: tt.load %b_ptrs, %mask_b, %b_zero
! TTIR: tt.dot
! TTIR-SAME: inputPrecision = ieee
! TTIR: tt.return
! JSON: "name": "fnacc_kernel_0"
! JSON: "matmul_precision": "ieee"

subroutine precision_ieee(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) matmul_precision(ieee)
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

! FIR-LABEL: func.func @_QPprecision_ieee
! FIR: fnacc.matmul_precision = "ieee"
! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: tt.load %a_ptrs, %mask_a, %a_zero
! TTIR: tt.load %b_ptrs, %mask_b, %b_zero
! TTIR: tt.dot
! TTIR-SAME: inputPrecision = ieee
! TTIR: tt.return
! JSON: "name": "fnacc_kernel_1"
! JSON: "matmul_precision": "ieee"

subroutine precision_tf32(a,b,c)
  real :: a(:,:), b(:,:), c(:,:)
  real :: acc
  integer :: i,j,p
  !$fnacc parallel tile(64,64,32) matmul_precision(tf32)
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

! FIR-LABEL: func.func @_QPprecision_tf32
! FIR: fnacc.matmul_precision = "tf32"
! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR: tt.load %a_ptrs, %mask_a, %a_zero
! TTIR: tt.load %b_ptrs, %mask_b, %b_zero
! TTIR: tt.dot
! TTIR-SAME: inputPrecision = tf32
! TTIR: tt.return
! JSON: "name": "fnacc_kernel_2"
! JSON: "matmul_precision": "tf32"

subroutine precision_tf32x3(a,b,c)
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

! FIR-LABEL: func.func @_QPprecision_tf32x3
! FIR: fnacc.matmul_precision = "tf32x3"
! TTIR-LABEL: tt.func @fnacc_kernel_3(
! TTIR: tt.load %a_ptrs, %mask_a, %a_zero
! TTIR: tt.load %b_ptrs, %mask_b, %b_zero
! TTIR: tt.dot
! TTIR-SAME: inputPrecision = tf32x3
! TTIR: tt.return
! JSON: "name": "fnacc_kernel_3"
! JSON: "matmul_precision": "tf32x3"

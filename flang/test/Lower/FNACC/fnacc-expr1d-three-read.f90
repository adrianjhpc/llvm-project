! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine expr1d_three_read(n, a, b, d, c)
  integer :: n
  real :: a(n), b(n), d(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + b(i) + d(i)
  end do
end subroutine

! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %read1: !tt.ptr<f32>
! TTIR-SAME: %read2: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "name": "read0"
! JSON: "name": "read1"
! JSON: "name": "read2"
! JSON: "name": "write"


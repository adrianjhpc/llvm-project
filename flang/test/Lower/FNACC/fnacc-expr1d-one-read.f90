! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine expr1d_one_read(n, a, c)
  integer :: n
  real :: a(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + 1.0
  end do
end subroutine

! HOST: call @__fnacc_launch_f32_v1

! TTIR: tt.func @fnacc_kernel_0
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR: tt.store

! JSON: "kind": "expr1d"
! JSON: "role": "read"
! JSON: "name": "read0"
! JSON: "role": "write"
! JSON: "name": "write"


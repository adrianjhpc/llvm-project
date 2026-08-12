! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine vector_add_f64(n, a, b, c)
  integer :: n
  real(8) :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine

! HOST: call @__fnacc_launch_f64_v1

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %b: !tt.ptr<f64>
! TTIR-SAME: %c: !tt.ptr<f64>
! TTIR-SAME: %n: i32

! TTIR: tt.load
! TTIR-SAME: tensor<128x!tt.ptr<f64>>

! TTIR: tt.load
! TTIR-SAME: tensor<128x!tt.ptr<f64>>

! TTIR: arith.addf
! TTIR-SAME: tensor<128xf64>

! TTIR: tt.store
! TTIR-SAME: tensor<128x!tt.ptr<f64>>

! JSON: "type": "ptr<f64>"


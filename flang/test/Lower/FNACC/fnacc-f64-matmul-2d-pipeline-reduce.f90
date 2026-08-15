! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json f64-matmul-strategy=reduce" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine matmul_2d_f64_fnacc_kernel(a, b, c)
  real(8) :: a(:, :), b(:, :), c(:, :)
  integer :: i, j, p
  real(8) :: acc

  !$fnacc parallel tile(8, 8)
  do j = 1, size(c, 2)
    do i = 1, size(c, 1)
      acc = 0.0_8
      do p = 1, size(a, 2)
        acc = acc + a(i, p) * b(p, j)
      end do
      c(i, j) = acc
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_launch_matmul_f64_v1

! HOST-LABEL: func.func @_QPmatmul_2d_f64_fnacc_kernel
! HOST: call @__fnacc_launch_matmul_f64_v1
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %b: !tt.ptr<f64>
! TTIR-SAME: %c: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR-SAME: %k: i32

! TTIR: scf.for
! TTIR: tt.load
! TTIR-SAME: !tt.ptr<f64>
! TTIR: arith.mulf
! TTIR-SAME: tensor<8x8x8xf64>
! TTIR: tt.reduce
! TTIR: arith.addf %{{.*}}, %{{.*}} : f64
! TTIR: }) {axis = 2 : i32} : (tensor<8x8x8xf64>) -> tensor<8x8xf64>
! TTIR: arith.addf %{{.*}}, %{{.*}} : tensor<8x8xf64>
! TTIR: tt.store
! TTIR-SAME: !tt.ptr<f64>

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "id": 0
! JSON-DAG: "name": "fnacc_kernel_0"
! JSON-DAG: "kind": "matmul2d"
! JSON-DAG: "rank": 2
! JSON-DAG: "tile": [8, 8, 8]
! JSON-DAG: "type": "ptr<f64>"
! JSON-DAG: "role": "extent_k"


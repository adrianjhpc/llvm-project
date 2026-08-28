! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json f64-matmul-strategy=dot" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine matmul_2d_f64_fnacc_dot_strategy(a, b, c)
  real(8) :: a(:, :), b(:, :), c(:, :)
  integer :: i, j, p
  real(8) :: acc

  ! Use a small K tile suitable for experimental FP64 tt.dot / MMA lowering.
  !$fnacc parallel tile(8, 8, 4)
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

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2

! HOST-LABEL: func.func @_QPmatmul_2d_f64_fnacc_dot_strategy
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-3: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %b: !tt.ptr<f64>
! TTIR-SAME: %c: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR-SAME: %k: i32

! TTIR: tt.make_range {start = 0 : i32, end = 8 : i32}
! TTIR: tt.make_range {start = 0 : i32, end = 8 : i32}
! TTIR: tt.make_range {start = 0 : i32, end = 4 : i32}

! TTIR: scf.for
! TTIR: tt.load
! TTIR-SAME: tensor<8x4x!tt.ptr<f64>>
! TTIR: tt.load
! TTIR-SAME: tensor<4x8x!tt.ptr<f64>>
! TTIR: tt.dot
! TTIR-SAME: tensor<8x4xf64>
! TTIR-SAME: tensor<4x8xf64>
! TTIR-SAME: tensor<8x8xf64>
! TTIR: scf.yield
! TTIR-SAME: tensor<8x8xf64>

! TTIR: tt.store
! TTIR-SAME: tensor<8x8x!tt.ptr<f64>>
! TTIR: tt.return

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "id": 0
! JSON-DAG: "name": "fnacc_kernel_0"
! JSON-DAG: "kind": "matmul2d"
! JSON-DAG: "launch_abi_version": 2
! JSON-DAG: "array_count": 3
! JSON-DAG: "output_count": 1
! JSON-DAG: "rank": 2
! JSON-DAG: "tile": [8, 8, 4]
! JSON-DAG: "type": "ptr<f64>"
! JSON-DAG: "role": "extent_k"

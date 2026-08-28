! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine matmul_2d_fnacc_kernel(a, b, c)
  real :: a(:, :), b(:, :), c(:, :)
  integer :: i, j, p
  real :: acc

  !$fnacc parallel tile(16, 16)
  do j = 1, size(c, 2)
    do i = 1, size(c, 1)
      acc = 0.0
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

! HOST-LABEL: func.func @_QPmatmul_2d_fnacc_kernel
! HOST-DAG: %[[BX:.*]] = arith.constant 16 : i32
! HOST-DAG: %[[BY:.*]] = arith.constant 16 : i32
! HOST-DAG: %[[BK:.*]] = arith.constant 32 : i32
! HOST: call @__fnacc_begin_launch_v2(%{{.*}}, %{{.*}}, %[[BX]], %[[BY]], %[[BK]],
! HOST-COUNT-3: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f32>
! TTIR-SAME: %b: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR-SAME: %k: i32

! TTIR: tt.make_range {start = 0 : i32, end = 16 : i32}
! TTIR: tt.make_range {start = 0 : i32, end = 16 : i32}
! TTIR: tt.make_range {start = 0 : i32, end = 32 : i32}

! TTIR: scf.for
! TTIR: tt.load
! TTIR-SAME: tensor<16x32x!tt.ptr<f32>>
! TTIR: tt.load
! TTIR-SAME: tensor<32x16x!tt.ptr<f32>>
! TTIR: tt.dot
! TTIR-SAME: tensor<16x32xf32>
! TTIR-SAME: tensor<32x16xf32>
! TTIR-SAME: tensor<16x16xf32>
! TTIR: scf.yield
! TTIR-SAME: tensor<16x16xf32>

! TTIR: tt.store
! TTIR-SAME: tensor<16x16x!tt.ptr<f32>>
! TTIR: tt.return

! JSON: "fnacc_schema_version": 1
! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "kind": "matmul2d"
! JSON: "rank": 2
! JSON: "tile": [16, 16, 32]
! JSON: "launch_abi_version": 2
! JSON: "array_count": 3
! JSON: "scalar_count": 0
! JSON: "output_count": 1
! JSON: "role": "extent_k"

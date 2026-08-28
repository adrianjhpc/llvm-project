! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine initialise_chunk_affine_1d(x_min, x_max, xmin, dx, &
                                      vertexx, vertexdx, cellx)
  implicit none
  integer :: x_min, x_max
  real(8) :: xmin, dx
  real(8) :: vertexx(x_min-2:x_max+3)
  real(8) :: vertexdx(x_min-2:x_max+3)
  real(8) :: cellx(x_min-2:x_max+2)
  integer :: j

  !$fnacc parallel tile(128)
  do j = x_min-2, x_max+3
    vertexx(j) = xmin + dx * float(j-x_min)
  end do

  !$fnacc parallel tile(128)
  do j = x_min-2, x_max+3
    vertexdx(j) = dx
  end do

  !$fnacc parallel tile(128)
  do j = x_min-2, x_max+2
    cellx(j) = 0.5_8 * (vertexx(j) + vertexx(j+1))
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f64_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_i32_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPinitialise_chunk_affine_1d
! HOST: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST-COUNT-2: call @__fnacc_bind_scalar_f64_v2
! HOST: call @__fnacc_bind_scalar_i32_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_bind_scalar_f64_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-2: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %scalar1: f64
! TTIR-SAME: %index0: i32
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-SAME: %array0_lower0: i32
! TTIR-SAME: %array0_stride0: i32
! TTIR: %source_x = arith.addi
! TTIR: arith.sitofp
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR: tt.store

! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-COUNT-2: tt.load
! TTIR: tt.store

! JSON: "kind": "multi_expr1d"
! JSON: "rank": 1
! JSON: "launch_abi_version": 2
! JSON: "kind": "multi_expr1d"
! JSON: "rank": 1
! JSON: "launch_abi_version": 2
! JSON: "kind": "multi_expr1d"
! JSON: "rank": 1
! JSON: "launch_abi_version": 2


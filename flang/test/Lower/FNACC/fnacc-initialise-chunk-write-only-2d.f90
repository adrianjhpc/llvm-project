! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine initialise_chunk_write_only_2d(x_min, x_max, y_min, y_max, &
                                          dx, dy, v)
  implicit none
  integer :: x_min, x_max, y_min, y_max
  real(8) :: dx, dy
  real(8) :: v(x_min-2:x_max+2, y_min-2:y_max+2)
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = y_min-2, y_max+2
    do j = x_min-2, x_max+2
      v(j, k) = dx * dy
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f64_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPinitialise_chunk_write_only_2d
! HOST: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST-COUNT-2: call @__fnacc_bind_scalar_f64_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %scalar1: f64
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-SAME: %loop_lower_y: i32
! TTIR-NOT: tt.load
! TTIR: arith.mulf
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 1
! JSON: "scalar_count": 2
! JSON: "output_count": 1


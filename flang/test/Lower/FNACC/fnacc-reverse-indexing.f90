! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine reverse_indexing_kernel(x_min, x_max, y_min, y_max, left_xmin, &
                                left_xmax, d, work, work0)
  implicit none
  integer :: x_min, x_max, y_min, y_max
  integer :: left_xmin, left_xmax, d
  real(8) :: work(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: work0(left_xmin-2:left_xmax+2, &
                           y_min-2:y_max+2)
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = y_min-d, y_max+d
    do j = 1, d
      work(x_min-j, k) = work0(left_xmax+1-j, k)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_i32_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPreverse_indexing_kernel
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-2: call @__fnacc_bind_array_v2
! HOST-COUNT-2: call @__fnacc_bind_scalar_i32_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %index0: i32
! TTIR-SAME: %index1: i32
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR: %access0_dim0_reversed = arith.subi
! TTIR: tt.load
! TTIR: %output0_dim0_reversed = arith.subi
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 2
! JSON: "scalar_count": 2
! JSON: "output_count": 1

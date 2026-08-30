! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine reduction_min_1d_bounds(lo, hi, a, result)
  implicit none
  integer :: lo, hi, i
  real(8) :: a(lo-2:hi+2), result

  result = huge(result)
  !$fnacc parallel tile(256) reduction(min:result)
  do i = lo, hi
    result = min(result, a(i))
  end do
end subroutine

subroutine reduction_min_2d_bounds(x_min, x_max, y_min, y_max, a, result)
  implicit none
  integer :: x_min, x_max, y_min, y_max, j, k
  real(8) :: a(x_min-2:x_max+2, y_min-2:y_max+2), result

  result = huge(result)
  !$fnacc parallel tile(16,16) reduction(min:result)
  do k = y_min, y_max
    do j = x_min, x_max
      result = min(result, a(j,k))
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_reduction_result_f64_v2
! HOST-DAG: func.func private @__fnacc_bind_reduction_result_f64_at_v2
! HOST-LABEL: func.func @_QPreduction_min_1d_bounds
! HOST: arith.maxsi
! HOST: call @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPreduction_min_2d_bounds
! HOST-COUNT-2: arith.maxsi
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %partials: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-SAME: %array0_lower0: i32
! TTIR-SAME: %array0_stride0: i32
! TTIR: %source_x = arith.addi %offs, %loop_lower_x_s
! TTIR: %vals_index = arith.subi %source_x, %vals_lower_s
! TTIR: %vals_offset = arith.muli %vals_index, %vals_stride_s
! TTIR: arith.minimumf %lhs, %rhs : f64

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %partials: !tt.ptr<f64>
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-SAME: %loop_lower_y: i32
! TTIR-SAME: %array0_lower0: i32
! TTIR-SAME: %array0_lower1: i32
! TTIR: %source_x = arith.addi %ix0, %loop_lower_x_s
! TTIR: %source_y = arith.addi %iy0, %loop_lower_y_s
! TTIR: %access0_x = arith.subi %source_x, %access0_lower0_s
! TTIR: %access0_y = arith.subi %source_y, %access0_lower1_s
! TTIR-COUNT-1: "tt.reduce"
! TTIR: arith.minimumf %lhs0, %rhs0 : f64

! JSON: "kind": "reduction_min1d"
! JSON: "rank": 1
! JSON: "output_count": 1
! JSON: "role": "loop_lower_x"
! JSON: "role": "array_lower_bound"
! JSON: "role": "array_stride"
! JSON: "kind": "reduction_multi2d"
! JSON: "rank": 2
! JSON: "output_count": 1
! JSON: "reduction_op": "min"

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_integer_sum(n, a, result)
  integer :: n, i, a(n), result
  result = 0
  !$fnacc parallel tile(256) reduction(+:result)
  do i = 1, n
    result = result + a(i)
  end do
end subroutine

subroutine fnacc_integer_product(n, a, result)
  integer :: n, i, a(n), result
  result = 1
  !$fnacc parallel tile(256) reduction(*:result)
  do i = 1, n
    result = result * a(i)
  end do
end subroutine

subroutine fnacc_integer_min(n, a, result)
  integer :: n, i, a(n), result
  result = huge(result)
  !$fnacc parallel tile(256) reduction(min:result)
  do i = 1, n
    result = min(result, a(i))
  end do
end subroutine

subroutine fnacc_integer_max_i64(n, a, result)
  integer :: n, i
  integer(8) :: a(n), result
  result = -huge(result)
  !$fnacc parallel tile(256) reduction(max:result)
  do i = 1, n
    result = max(result, a(i))
  end do
end subroutine

! HOST-COUNT-3: call @__fnacc_launch_reduce_i32_v2
! HOST: call @__fnacc_launch_reduce_i64_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: !tt.ptr<i32>
! TTIR: arith.constant 0 : i32
! TTIR: arith.addi %lhs, %rhs : i32

! TTIR-LABEL: tt.func @fnacc_kernel_1
! TTIR-SAME: !tt.ptr<i32>
! TTIR: arith.constant 1 : i32
! TTIR: arith.muli %lhs, %rhs : i32

! TTIR-LABEL: tt.func @fnacc_kernel_2
! TTIR-SAME: !tt.ptr<i32>
! TTIR: arith.constant 2147483647 : i32
! TTIR: arith.minsi %lhs, %rhs : i32

! TTIR-LABEL: tt.func @fnacc_kernel_3
! TTIR-SAME: !tt.ptr<i64>
! TTIR: arith.constant -9223372036854775808 : i64
! TTIR: arith.maxsi %lhs, %rhs : i64

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "kind": "reduction_sum1d"
! JSON-DAG: "kind": "reduction_product1d"
! JSON-DAG: "kind": "reduction_min1d"
! JSON-DAG: "kind": "reduction_max1d"
! JSON-DAG: "type": "ptr<i32>"
! JSON-DAG: "type": "ptr<i64>"

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_product_reduce(n, a, result)
  integer :: n, i
  real :: a(n), result

  result = 1.0
  !$fnacc parallel tile(256) reduction(*:result)
  do i = 1, n
    result = result * a(i)
  end do
end subroutine

subroutine fnacc_min_reduce(n, a, result)
  integer :: n, i
  real :: a(n), result

  result = huge(result)
  !$fnacc parallel tile(256) reduction(min:result)
  do i = 1, n
    result = min(result, a(i))
  end do
end subroutine

subroutine fnacc_max_reduce(n, a, result)
  integer :: n, i
  real :: a(n), result

  result = -huge(result)
  !$fnacc parallel tile(256) reduction(max:result)
  do i = 1, n
    result = max(result, a(i))
  end do
end subroutine

! HOST: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_bind_reduction_result_f32_v2
! HOST-COUNT-3: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR: arith.constant 1.000000e+00 : f32
! TTIR: arith.mulf %lhs, %rhs : f32
! TTIR-LABEL: tt.func @fnacc_kernel_1
! TTIR: arith.constant 0x7F800000 : f32
! TTIR: arith.minimumf %lhs, %rhs : f32
! TTIR-LABEL: tt.func @fnacc_kernel_2
! TTIR: arith.constant 0xFF800000 : f32
! TTIR: arith.maximumf %lhs, %rhs : f32

! JSON: "fnacc_schema_version": 1
! JSON: "kind": "reduction_product1d"
! JSON: "launch_abi_version": 2
! JSON: "reduction_op": "multiply"
! JSON: "kind": "reduction_min1d"
! JSON: "launch_abi_version": 2
! JSON: "reduction_op": "min"
! JSON: "kind": "reduction_max1d"
! JSON: "launch_abi_version": 2
! JSON: "reduction_op": "max"
! JSON: "kind": "reduction_stage1d"

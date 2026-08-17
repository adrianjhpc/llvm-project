! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_sum_reduce_first(n, a, sum)
  integer :: n
  real :: a(n), sum
  integer :: i

  sum = 0.0
  !$fnacc parallel tile(256) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i)
  end do
end subroutine

subroutine fnacc_dot_reduce_second(n, a, b, sum)
  integer :: n
  real(8) :: a(n), b(n), sum
  integer :: i

  sum = 0.0_8
  !$fnacc parallel tile(1024) reduction(+:sum)
  do i = 1, n
    sum = sum + a(i) * b(i)
  end do
end subroutine

! HOST-COUNT-2: call @__fnacc_launch_reduce_
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR: tt.reduce
! TTIR-LABEL: tt.func @fnacc_kernel_0_reduce_stage(
! TTIR: tensor<256xf32>
! TTIR: tt.reduce
! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR: tt.reduce
! TTIR-LABEL: tt.func @fnacc_kernel_1_reduce_stage(
! TTIR: tensor<1024xf64>
! TTIR: tt.reduce

! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "reduction_stage_id": 2
! JSON: "id": 2
! JSON: "name": "fnacc_kernel_0_reduce_stage"
! JSON: "kind": "reduction_stage1d"
! JSON: "id": 1
! JSON: "name": "fnacc_kernel_1"
! JSON: "reduction_stage_id": 3
! JSON: "id": 3
! JSON: "name": "fnacc_kernel_1_reduce_stage"
! JSON: "kind": "reduction_stage1d"

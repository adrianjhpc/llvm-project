! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.default.ttir json-output=%t.default.json" %t.fir -o %t.default.host.fir
! RUN: FileCheck %s --check-prefix=DEFAULT < %t.default.json
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.warp4.ttir json-output=%t.warp4.json num-warps=4" %t.fir -o %t.warp4.host.fir
! RUN: FileCheck %s --check-prefix=MULTI < %t.warp4.json
! RUN: FileCheck %s --check-prefix=TTIR < %t.warp4.ttir

subroutine fnacc_reduction_multiwarp(n, a, total)
  integer, intent(in) :: n
  real, intent(in) :: a(n)
  real, intent(out) :: total
  integer :: i

  total = 0.0

  !$fnacc parallel tile(256) reduction(+:total)
  do i = 1, n
    total = total + a(i)
  end do
end subroutine

! DEFAULT:      "name": "fnacc_kernel_0"
! DEFAULT:      "kind": "reduction_sum1d"
! DEFAULT:      "num_warps": 1
! DEFAULT:      "cuda_threads_per_cta": 32
! DEFAULT:      "name": "fnacc_kernel_0_reduce_stage"
! DEFAULT:      "kind": "reduction_stage1d"
! DEFAULT:      "num_warps": 1
! DEFAULT:      "cuda_threads_per_cta": 32

! MULTI:        "name": "fnacc_kernel_0"
! MULTI:        "kind": "reduction_sum1d"
! MULTI:        "num_warps": 4
! MULTI:        "cuda_threads_per_cta": 128
! MULTI:        "name": "fnacc_kernel_0_reduce_stage"
! MULTI:        "kind": "reduction_stage1d"
! MULTI:        "num_warps": 4
! MULTI:        "cuda_threads_per_cta": 128

! TTIR: module attributes {{.*}}"ttg.num-warps" = 4 : i32
! TTIR: tt.func @fnacc_kernel_0(
! TTIR: tt.func @fnacc_kernel_0_reduce_stage(

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine fnacc_add_allocatable_f64(n, a, b, c)
  use, intrinsic :: iso_fortran_env, only : int64, real64
  integer(kind=int64), intent(in) :: n
  real(kind=real64), allocatable, intent(in) :: a(:), b(:)
  real(kind=real64), allocatable, intent(inout) :: c(:)
  integer(kind=int64) :: i

  !$fnacc parallel tile(256)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2

! HOST-LABEL: func.func @_QPfnacc_add_allocatable_f64
! HOST: %[[N64:.*]] = fir.load {{.*}} : !fir.ref<i64>
! HOST: %[[N32:.*]] = fir.convert %[[N64]] : (i64) -> i32
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-3: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0
! TTIR-SAME: %a: !tt.ptr<f64>
! TTIR-SAME: %b: !tt.ptr<f64>
! TTIR-SAME: %c: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR: arith.addf
! TTIR-SAME: f64
! TTIR: tt.store

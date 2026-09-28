! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --TileOffload-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine tileoff_expr_too_many_scalars(n, a, c, s0, s1, s2, s3)
  integer :: n
  real :: a(n), c(n)
  real :: s0, s1, s2, s3
  integer :: i

  !$tileoff parallel tile(128)
  do i = 1, n
    c(i) = a(i) + s0 + s1 + s2 + s3
  end do
end subroutine

! HOST-LABEL: func.func @_QPtileoff_expr_too_many_scalars
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-2: call @__tileoff_bind_array_v2
! HOST-COUNT-4: call @__tileoff_bind_scalar_f32_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST-NOT: TileOffload.launch

! TTIR: tt.func @tileoff_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %scalar0: f32
! TTIR-SAME: %scalar1: f32
! TTIR-SAME: %scalar2: f32
! TTIR-SAME: %scalar3: f32
! TTIR-SAME: %n: i32
! TTIR: tt.splat %scalar3
! TTIR: tt.store

! JSON: "kind": "expr1d"
! JSON: "launch_abi_version": 2
! JSON: "array_count": 2
! JSON: "scalar_count": 4
! JSON: "output_count": 1
! JSON: "name": "scalar3"
! JSON-SAME: "scalar_index": 3


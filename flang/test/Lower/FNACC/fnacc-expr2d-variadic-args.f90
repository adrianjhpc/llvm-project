! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine variadic_expr2d(n, m, a, b, c, d, out, s0, s1, s2, s3)
  integer :: n, m
  real :: a(n,m), b(n,m), c(n,m), d(n,m), out(n,m)
  real :: s0, s1, s2, s3
  integer :: i, j

  !$fnacc parallel tile(16, 16)
  do j = 1, m
    do i = 1, n
      out(i,j) = a(i,j) * s0 + b(i,j) * s1 + &
                 c(i,j) * s2 + d(i,j) * s3
    end do
  end do
end subroutine

! HOST-LABEL: func.func @_QPvariadic_expr2d
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-5: call @__fnacc_bind_array_v2
! HOST-COUNT-4: call @__fnacc_bind_scalar_f32_v2
! HOST: call @__fnacc_commit_launch_v2

! TTIR: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read3: !tt.ptr<f32>
! TTIR-SAME: %scalar3: f32
! TTIR-SAME: %n: i32
! TTIR-SAME: %m: i32
! TTIR: tt.splat %read3
! TTIR: tt.splat %scalar3
! TTIR: tt.store

! JSON: "kind": "expr2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 5
! JSON: "scalar_count": 4
! JSON: "output_count": 1

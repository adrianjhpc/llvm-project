! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr1d_neg(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = -a(i)
  end do
end subroutine

subroutine fnacc_expr1d_abs(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = abs(a(i))
  end do
end subroutine

subroutine fnacc_expr1d_sqrt(n, a, b)
  integer :: n
  real :: a(n), b(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    b(i) = sqrt(a(i))
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-COUNT-3: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: arith.negf
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store
! TTIR: tt.return

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: math.absf
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store
! TTIR: tt.return

! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR-SAME: %read0: !tt.ptr<f32>
! TTIR-SAME: %c: !tt.ptr<f32>
! TTIR-SAME: %n: i32
! TTIR: math.sqrt
! TTIR-SAME: tensor<128xf32>
! TTIR: tt.store
! TTIR: tt.return

! JSON: "fnacc_schema_version": 1

! JSON: "id": 0
! JSON: "name": "fnacc_kernel_0"
! JSON: "kind": "expr1d"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]

! JSON: "id": 1
! JSON: "name": "fnacc_kernel_1"
! JSON: "kind": "expr1d"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]

! JSON: "id": 2
! JSON: "name": "fnacc_kernel_2"
! JSON: "kind": "expr1d"
! JSON: "rank": 1
! JSON: "tile": [128, 1, 1]

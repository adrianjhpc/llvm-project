! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --tileoff-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine tileoff_expr1d_unary_f64(n, a, b)
  integer :: n
  real(8) :: a(n), b(n)
  integer :: i

  !$tileoff parallel tile(128)
  do i = 1, n
    b(i) = sqrt(abs(-a(i)))
  end do
end subroutine

! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_bind_array_v2
! HOST-DAG: func.func private @__tileoff_commit_launch_v2
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-2: call @__tileoff_bind_array_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST-NOT: TileOffload.launch

! TTIR-LABEL: tt.func @tileoff_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<f64>
! TTIR-SAME: %c: !tt.ptr<f64>
! TTIR-SAME: %n: i32
! TTIR: arith.negf
! TTIR-SAME: tensor<128xf64>
! TTIR: math.absf
! TTIR-SAME: tensor<128xf64>
! TTIR: math.sqrt
! TTIR-SAME: tensor<128xf64>
! TTIR: tt.store

! JSON: "tileoff_schema_version": 1
! JSON: "kind": "expr1d"
! JSON: "type": "ptr<f64>"

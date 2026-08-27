! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine fnacc_expr_integer_i32(n, a, b, c, alpha)
  integer :: n, alpha
  integer :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = merge((a(i) + alpha) * b(i), a(i) / b(i), a(i) > b(i))
  end do
end subroutine

subroutine fnacc_expr_integer_i64(n, a, b, c)
  integer :: n
  integer(8) :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = min(a(i), b(i))
  end do
end subroutine

subroutine fnacc_expr_integer_i8(n, a, b, c)
  integer :: n
  integer(1) :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = abs(a(i)) + b(i)
  end do
end subroutine

subroutine fnacc_expr_integer_i16(n, a, b, c)
  integer :: n
  integer(2) :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128)
  do i = 1, n
    c(i) = a(i) - b(i)
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-COUNT-4: call @__fnacc_begin_launch_v2
! HOST: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %read0: !tt.ptr<i32>
! TTIR-SAME: %read1: !tt.ptr<i32>
! TTIR-SAME: %c: !tt.ptr<i32>
! TTIR-SAME: %scalar0: i32
! TTIR-DAG: arith.addi
! TTIR-DAG: arith.muli
! TTIR-DAG: arith.divsi
! TTIR-DAG: arith.cmpi sgt
! TTIR: arith.select

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %read0: !tt.ptr<i64>
! TTIR-SAME: %read1: !tt.ptr<i64>
! TTIR-SAME: %c: !tt.ptr<i64>
! TTIR: arith.minsi

! TTIR-LABEL: tt.func @fnacc_kernel_2(
! TTIR-SAME: %read0: !tt.ptr<i8>
! TTIR-SAME: %read1: !tt.ptr<i8>
! TTIR-SAME: %c: !tt.ptr<i8>
! TTIR: arith.subi
! TTIR: arith.cmpi slt
! TTIR: arith.select
! TTIR: arith.addi

! TTIR-LABEL: tt.func @fnacc_kernel_3(
! TTIR-SAME: %a: !tt.ptr<i16>
! TTIR-SAME: %b: !tt.ptr<i16>
! TTIR-SAME: %c: !tt.ptr<i16>
! TTIR: arith.subi

! JSON: "fnacc_schema_version": 1
! JSON-DAG: "type": "ptr<i32>"
! JSON-DAG: "type": "i32"
! JSON-DAG: "type": "ptr<i64>"
! JSON-DAG: "type": "ptr<i8>"
! JSON-DAG: "type": "ptr<i16>"

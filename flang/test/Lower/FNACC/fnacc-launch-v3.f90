! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 ttir-output=%t.v3.ttir json-output=%t.v3.json" %t.fir -o %t.v3.fir
! RUN: FileCheck %s --check-prefix=V3 --implicit-check-not=__fnacc_begin_launch_v2 --implicit-check-not=__fnacc_bind_ --implicit-check-not=__fnacc_commit_launch_v2 --input-file=%t.v3.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.v2.ttir json-output=%t.v2.json" %t.fir -o %t.v2.fir
! RUN: FileCheck %s --check-prefix=V2 --implicit-check-not=__fnacc_launch_v3 --input-file=%t.v2.fir
! RUN: diff %t.v2.ttir %t.v3.ttir
! RUN: diff %t.v2.json %t.v3.json
! RUN: fir-opt --fnacc-lower-to-runtime="launch-abi=3" %t.fir -o %t.direct.fir
! RUN: FileCheck %s --check-prefix=V3 --input-file=%t.direct.fir
! RUN: not fir-opt --fnacc-lower-to-runtime="launch-abi=4" %t.fir 2>&1 | FileCheck %s --check-prefix=BAD
! BAD: FNACC launch-abi must be 2 or 3

subroutine scaled_add(n, a, b, c, scale)
  integer :: n, i
  real(8) :: a(n), b(n), c(n), scale
  !$fnacc parallel tile(256)
  do i=1,n
    c(i)=a(i)+scale*b(i)
  enddo
end subroutine
! V3-LABEL: func.func @_QPscaled_add(
! V3: fir.alloca
! V3: fir.insert_value
! V3: call @__fnacc_launch_v3
! V3-NOT: call @__fnacc_launch_v3
! V3: return
! V2-LABEL: func.func @_QPscaled_add(
! V2: call @__fnacc_begin_launch_v2
! V2: call @__fnacc_bind_array_v2
! V2: call @__fnacc_bind_scalar_f64_v2
! V2: call @__fnacc_commit_launch_v2

subroutine dot(n,a,b,s)
  integer :: n,i
  real :: a(n),b(n),s
  !$fnacc parallel tile(256) reduction(+:s)
  do i=1,n
    s=s+a(i)*b(i)
  enddo
end subroutine
! V3-LABEL: func.func @_QPdot(
! V3: call @__fnacc_launch_v3
! V3-NOT: call @__fnacc_launch_v3
! V3: return
! V2-LABEL: func.func @_QPdot(
! V2: call @__fnacc_begin_launch_v2
! V2: call @__fnacc_bind_reduction_result_f32
! V2: call @__fnacc_commit_launch_v2

subroutine multi(lo,hi,a,s,t)
  integer :: lo,hi,i,j
  real(8) :: a(lo:hi,lo:hi),s,t
  !$fnacc parallel tile(16,16) reduction(+:s,+:t)
  do j=lo,hi
    do i=lo,hi
      s=s+a(i,j)
      t=t+a(i,j)*a(i,j)
    enddo
  enddo
end subroutine
! V3-LABEL: func.func @_QPmulti(
! V3: call @__fnacc_launch_v3
! V3-NOT: call @__fnacc_launch_v3
! V3: return
! V2-LABEL: func.func @_QPmulti(
! V2: call @__fnacc_begin_launch_v2
! V2: call @__fnacc_bind_reduction_result_f64_at_v2
! V2: call @__fnacc_bind_reduction_result_f64_at_v2
! V2: call @__fnacc_commit_launch_v2

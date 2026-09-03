! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine store_chain_kernel(xmin, xmax, ymin, ymax, ht, &
                              d, v, xa, ya,       &
                              p, vs, xv0, yv0,   &
                              xv1, yv1)
  implicit none
  integer :: xmin, xmax, ymin, ymax
  real(8) :: ht
  real(8) :: d(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: v(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: xa(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: ya(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: p(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: vs(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: xv0(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: yv0(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: xv1(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: yv1(xmin-2:xmax+3, ymin-2:ymax+3)
  real(8) :: a
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = ymin, ymax + 1
    do j = xmin, xmax + 1
      a = ht / ((d(j-1, k-1) * v(j-1, k-1) + &
                                d(j,   k-1) * v(j,   k-1) + &
                                d(j,   k)   * v(j,   k)   + &
                                d(j-1, k)   * v(j-1, k)) * &
                               0.25_8)

      xv1(j, k) = xv0(j, k) - a * &
        (xa(j, k)   * (p(j, k)   - p(j-1, k)) + &
         xa(j, k-1) * (p(j, k-1) - p(j-1, k-1)))
      yv1(j, k) = yv0(j, k) - a * &
        (ya(j, k)   * (p(j, k)   - p(j, k-1)) + &
         ya(j-1, k) * (p(j-1, k) - p(j-1, k-1)))

      xv1(j, k) = xv1(j, k) - a * &
        (xa(j, k)   * (vs(j, k)   - vs(j-1, k)) + &
         xa(j, k-1) * (vs(j, k-1) - vs(j-1, k-1)))
      yv1(j, k) = yv1(j, k) - a * &
        (ya(j, k)   * (vs(j, k)   - vs(j, k-1)) + &
         ya(j-1, k) * (vs(j-1, k) - vs(j-1, k-1)))
    end do
  end do
end subroutine

subroutine cross_array_store_chain(xmin, xmax, ymin, ymax, a, post, pre)
  implicit none
  integer :: xmin, xmax, ymin, ymax
  real(8) :: a(xmin:xmax, ymin:ymax)
  real(8) :: post(xmin:xmax, ymin:ymax)
  real(8) :: pre(xmin:xmax, ymin:ymax)
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = ymin, ymax
    do j = xmin, xmax
      post(j, k) = a(j, k) + 1.0_8
      pre(j, k) = post(j, k) * 2.0_8
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPstore_chain_kernel
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-10: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array9: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR-COUNT-2: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 10
! JSON: "scalar_count": 1
! JSON: "output_count": 2

! TTIR-LABEL: tt.func @fnacc_kernel_1(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %array2: !tt.ptr<f64>
! TTIR: %[[POST0:expr[0-9]+]] = arith.addf %access0_value,
! TTIR: tt.store {{.*}}, %[[POST0]], %mask
! TTIR: %[[PRE:expr[0-9]+]] = arith.mulf %[[POST0]],
! TTIR: tt.store {{.*}}, %[[PRE]], %mask
! TTIR-NOT: arith.mulf %access1_value

! JSON: "id": 1
! JSON: "kind": "stencil2d"
! JSON: "array_count": 3
! JSON: "output_count": 2

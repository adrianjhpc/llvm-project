! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine mixed_rank_kernel(x_min, x_max, y_min, y_max, dx, dy, &
                                     p, v)
  implicit none
  integer :: x_min, x_max, y_min, y_max
  real(8) :: dx(x_min-2:x_max+2)
  real(8) :: dy(y_min-2:y_max+2)
  real(8) :: p(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: v(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: px, py, dir, mag
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = y_min, y_max
    do j = x_min, x_max
      px = (p(j+1, k) - p(j-1, k)) / &
               (dx(j) + dx(j+1))
      py = (p(j, k+1) - p(j, k-1)) / &
               (dy(k) + dy(k+1))

      if (px >= 0.0_8 .or. py >= 0.0_8) then
        v(j, k) = 0.0_8
      else
        dir = 1.0_8
        if (py < 0.0_8) dir = -1.0_8
        px = dir* max(1.0e-16_8, abs(px))
        mag = sqrt(px**2 + py**2)
        v(j, k) = dx(j) * dy(k) * mag
      end if
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPmixed_rank_kernel
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-4: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array3: !tt.ptr<f64>
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR: arith.subi %source_x, %access{{[0-9]+}}_lower0_s
! TTIR: arith.subi %source_y, %access{{[0-9]+}}_lower0_s
! TTIR: arith.ori
! TTIR: math.sqrt
! TTIR: arith.select
! TTIR-COUNT-1: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 4
! JSON: "output_count": 1



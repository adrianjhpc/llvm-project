! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: python3 -m json.tool %t.json > /dev/null

subroutine stencil_like(x_min, x_max, y_min, y_max, dt, volume, &
                               xarea, xvel, pressure, density0, energy0, &
                               energy1, density1)
  implicit none
  integer :: x_min, x_max, y_min, y_max
  real(8) :: dt
  real(8) :: volume(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: xarea(x_min-2:x_max+3, y_min-2:y_max+2)
  real(8) :: xvel(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: pressure(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: density0(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: energy0(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: energy1(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: density1(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: left_flux, right_flux, total_flux, volume_change
  real(8) :: energy_change, dead_min_cell_volume
  integer :: j, k

  !$fnacc parallel tile(16, 16)
  do k = y_min, y_max
    do j = x_min, x_max
      left_flux = xarea(j, k) * (xvel(j, k) + xvel(j, k+1)) * dt * 0.5_8
      right_flux = xarea(j+1, k) * (xvel(j+1, k) + xvel(j+1, k+1)) * dt * 0.5_8
      total_flux = right_flux - left_flux
      volume_change = volume(j, k) / (volume(j, k) + total_flux)
      dead_min_cell_volume = min(volume(j, k) + total_flux, volume(j, k))
      energy_change = pressure(j, k) / density0(j, k) * total_flux / volume(j, k)
      energy1(j, k) = energy0(j, k) - energy_change
      density1(j, k) = density0(j, k) * volume_change
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f64_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPstencil_like
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-8: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_bind_scalar_f64_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array7: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR-SAME: %loop_lower_x: i32
! TTIR-SAME: %loop_lower_y: i32
! TTIR-SAME: %array0_lower0: i32
! TTIR-SAME: %array0_stride1: i32
! TTIR: arith.constant 1 : i32
! TTIR: arith.subi %{{.*}}, %{{.*}}_lower0_s
! TTIR-COUNT-2: tt.store

! JSON: "kind": "stencil2d"
! JSON: "launch_abi_version": 2
! JSON: "array_count": 8
! JSON: "scalar_count": 1
! JSON: "output_count": 2
! JSON: "role": "array_lower_bound"
! JSON: "role": "array_stride"

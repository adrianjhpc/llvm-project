! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine field_summary_multi_reduction(x_min, x_max, y_min, y_max, &
                                         volume, density0, energy0, pressure, &
                                         xvel0, yvel0,                       &
                                         vol, mass, press, ie, ke)
  implicit none
  integer :: x_min, x_max, y_min, y_max
  real(8) :: volume(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: density0(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: energy0(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: pressure(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: xvel0(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: yvel0(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: vol, mass, press, ie, ke
  real(8) :: vsqrd, cell_vol, cell_mass
  integer :: j, k, jv, kv

  !$fnacc parallel tile(16,16) reduction(+:vol, +:mass, +:press, +:ie, +:ke)
  do k = y_min, y_max
    do j = x_min, x_max
      vsqrd = 0.0_8
      do kv = k, k + 1
        do jv = j, j + 1
         vsqrd = vsqrd + 0.25_8 * &
              (xvel0(jv,kv)**2 + yvel0(jv,kv)**2)
        end do
      end do
      cell_vol = volume(j,k)
      cell_mass = cell_vol * density0(j,k)
      vol = vol + cell_vol
      mass = mass + cell_mass
      ie = ie + cell_mass * energy0(j,k)
      ke = ke + cell_mass * 0.5_8 * vsqrd
      press = press + cell_vol * pressure(j,k)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_reduction_result_f64_at_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPfield_summary_multi_reduction
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-6: call @__fnacc_bind_array_v2
! HOST-COUNT-5: call @__fnacc_bind_reduction_result_f64_at_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array5: !tt.ptr<f64>
! TTIR-SAME: %partials: !tt.ptr<f64>
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR-COUNT-5: "tt.reduce"
! TTIR: tt.store %reduction4_ptr
! TTIR-LABEL: tt.func @fnacc_kernel_0_reduce_stage(

! JSON: "kind": "reduction_multi2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 6
! JSON: "scalar_count": 0
! JSON: "output_count": 5
! JSON: "reduction_op": "add"


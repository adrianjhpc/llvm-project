! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine conditional_stencil_index(x_min, x_max, y_min, y_max, &
                                     flux, field, spacing, output)
  implicit none
  integer :: x_min, x_max, y_min, y_max, j, k
  integer :: donor, upwind, downwind, dif
  real(8) :: flux(x_min-2:x_max+3, y_min-2:y_max+2)
  real(8) :: field(x_min-2:x_max+3, y_min-2:y_max+2)
  real(8) :: spacing(x_min-2:x_max+3)
  real(8) :: output(x_min-2:x_max+2, y_min-2:y_max+2)

  !$fnacc parallel tile(16,16)
  do k = y_min, y_max
    do j = x_min, x_max+2
      if (flux(j,k) < 0.0_8) then
        upwind = j+2
        donor = j+1
        downwind = j
        dif = donor
      else
        upwind = j-1
        donor = j
        downwind = j+1
        dif = upwind
      end if
      output(j,k) = field(donor,k) + field(upwind,k) + &
                    field(downwind,k) + spacing(dif)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPconditional_stencil_index
! HOST-COUNT-4: call @__fnacc_bind_array_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %array2: !tt.ptr<f64>
! TTIR-SAME: %array3: !tt.ptr<f64>
! TTIR: %access0_value = tt.load
! TTIR: arith.cmpf olt, %access0_value
! TTIR: arith.select
! TTIR: tt.addptr
! TTIR-COUNT-4: tt.load
! TTIR-COUNT-3: arith.addf
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "array_count": 4
! JSON: "scalar_count": 0
! JSON: "output_count": 1

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json

! Model the final CloverLeaf momentum update.  The output array is also read
! at the same logical element and must occupy one read_write ABI slot.
subroutine stencil_readwrite(x_min, x_max, y_min, y_max, vel, mass_pre, &
                             mom_flux, mass_post)
  implicit none
  integer :: x_min, x_max, y_min, y_max, j, k
  real(8) :: vel(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: mass_pre(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: mom_flux(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: mass_post(x_min-2:x_max+3, y_min-2:y_max+3)

  !$fnacc parallel tile(16,16)
  do k = y_min, y_max+1
    do j = x_min, x_max+1
      vel(j,k) = (vel(j,k) * mass_pre(j,k) + mom_flux(j-1,k) - &
                  mom_flux(j,k)) / mass_post(j,k)
    end do
  end do
end subroutine

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR: %access2_dim0_delta = arith.constant -1 : i32
! TTIR: arith.mulf
! TTIR: arith.addf
! TTIR: arith.subf
! TTIR: arith.divf
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "array_count": 4
! JSON: "role": "read_write"
! JSON: "array_index": 0

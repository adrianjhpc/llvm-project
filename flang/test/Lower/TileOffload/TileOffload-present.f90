! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: FileCheck %s --check-prefix=FIR --input-file=%t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine present_explicit_shape(n, a, alpha)
  integer :: n
  real :: a(n), alpha

  !$tileoff present(a, alpha)
end subroutine

subroutine present_assumed_shape(a)
  real :: a(:)

  !$tileoff present(a)
end subroutine

! FIR-LABEL: func.func @_QPpresent_explicit_shape
! FIR-COUNT-2: TileOffload.present
! FIR-LABEL: func.func @_QPpresent_assumed_shape
! FIR: TileOffload.present

! HOST-DAG: func.func private @__tileoff_present_bytes
! HOST-DAG: func.func private @__tileoff_present_desc
! HOST-LABEL: func.func @_QPpresent_explicit_shape
! HOST-COUNT-2: call @__tileoff_present_bytes
! HOST-LABEL: func.func @_QPpresent_assumed_shape
! HOST: call @__tileoff_present_desc
! HOST-NOT: TileOffload.present

! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json emit-fortran-aliases=true" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --input-file=%t.host.fir

subroutine fnacc_alias_external_caller()
  interface
    subroutine ordinary_external_callee()
    end subroutine
  end interface

  !$fnacc wait
  call ordinary_external_callee()
end subroutine

! CHECK-NOT: @_QPordinary_external_callee
! CHECK-DAG: func.func private @ordinary_external_callee_()
! CHECK-DAG: fir.call @ordinary_external_callee_
! CHECK-NOT: @_QPordinary_external_callee

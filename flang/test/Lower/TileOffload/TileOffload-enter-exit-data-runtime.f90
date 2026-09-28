! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-runtime %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

subroutine test_tileoff_enter_exit_data_runtime(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$tileoff enter data copyin(a) create(c)
  !$tileoff enter data copyin(a, b)
  !$tileoff exit data copyout(a, b) delete(a, b)
  !$tileoff exit data copyout(a, c) delete(a, c)
end subroutine

! HOST-DAG: func.func private @__tileoff_enter_data_region
! HOST-DAG: func.func private @__tileoff_data_copyin_bytes
! HOST-DAG: func.func private @__tileoff_data_create_bytes
! HOST-DAG: func.func private @__tileoff_data_copyout_bytes
! HOST-DAG: func.func private @__tileoff_data_delete
! HOST-DAG: func.func private @__tileoff_exit_data_region

! HOST-LABEL: func.func @_QPtest_tileoff_enter_exit_data_runtime
! HOST: call @__tileoff_enter_data_region
! HOST: call @__tileoff_data_copyin_bytes
! HOST: call @__tileoff_data_create_bytes
! HOST: call @__tileoff_enter_data_region
! HOST: call @__tileoff_data_copyin_bytes
! HOST: call @__tileoff_data_copyin_bytes
! HOST: call @__tileoff_data_copyout_bytes
! HOST: call @__tileoff_data_copyout_bytes
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_exit_data_region
! HOST: call @__tileoff_data_copyout_bytes
! HOST: call @__tileoff_data_copyout_bytes
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_data_delete
! HOST: call @__tileoff_exit_data_region

! HOST-NOT: TileOffload.copyin
! HOST-NOT: TileOffload.create
! HOST-NOT: TileOffload.copyout
! HOST-NOT: TileOffload.delete

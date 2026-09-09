! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-runtime --canonicalize %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir

! Match calls within the user functions, not runtime declarations. The original
! omission from the lexical chain loses the directives following these branches.
subroutine enter_after_allocation(n, a, b, c)
  integer :: n, err
  real(8), allocatable :: a(:), b(:), c(:)
  allocate(a(n), b(n), c(n), stat=err)
  if (err /= 0) then
    print *, 'allocation failed', err
    stop 1
  endif
  !$fnacc enter data create(a, b, c)
end subroutine
! HOST-LABEL: func.func @_QPenter_after_allocation(
! HOST: call @__fnacc_enter_data_region
! HOST: call @__fnacc_data_create_desc
! HOST: call @__fnacc_data_create_desc
! HOST: call @__fnacc_data_create_desc

subroutine update_after_return(skip, a)
  logical :: skip
  real(8) :: a(4)
  if (skip) then
    return
  endif
  !$fnacc update device(a)
  !$fnacc update host(a)
end subroutine
! HOST-LABEL: func.func @_QPupdate_after_return(
! HOST: call @__fnacc_update_device_{{bytes|desc}}
! HOST: call @__fnacc_update_host_{{bytes|desc}}

subroutine exit_after_stop(fail, a)
  logical :: fail
  real(8) :: a(4)
  if (fail) then
    stop 2
  endif
  !$fnacc exit data delete(a)
end subroutine
! HOST-LABEL: func.func @_QPexit_after_stop(
! HOST: call @__fnacc_data_delete
! HOST: call @__fnacc_exit_data_region

subroutine wait_after_nested_stop(fail, inner)
  logical :: fail, inner
  if (fail) then
    if (inner) then
      stop 3
    endif
  endif
  !$fnacc wait
end subroutine
! HOST-LABEL: func.func @_QPwait_after_nested_stop(
! HOST: call @__fnacc_wait

! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_fnacc_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a) create(c)
  !$fnacc enter data copyin(a, b)
  !$fnacc exit data copyout(a, b) delete(a, b)
  !$fnacc exit data copyout(a, c) delete(a, c)
end subroutine

! CHECK: fnacc.data_region_enter
! CHECK: fnacc.copyin
! CHECK: fnacc.create
! CHECK: fnacc.data_region_enter
! CHECK: fnacc.copyin
! CHECK: fnacc.copyin
! CHECK: fnacc.copyout
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.delete
! CHECK: fnacc.data_region_exit
! CHECK: fnacc.copyout
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.delete
! CHECK: fnacc.data_region_exit

subroutine test_fnacc_enter_exit_data_components(chunk)
  type field_type
    real, allocatable :: density0(:)
    real, allocatable :: energy0(:)
  end type
  type tile_type
    type(field_type) :: field
  end type
  type chunk_type
    type(tile_type), allocatable :: tiles(:)
  end type
  type(chunk_type) :: chunk

  !$fnacc enter data &
  !$fnacc& copyin(chunk%tiles(1)%field%density0) &
  !$fnacc& create(chunk%tiles(1)%field%energy0)
  !$fnacc present(chunk%tiles(1)%field%density0)
  !$fnacc update device(chunk%tiles(1)%field%density0)
  !$fnacc update host(chunk%tiles(1)%field%density0)
  !$fnacc release(chunk%tiles(1)%field%density0)
  !$fnacc exit data &
  !$fnacc& copyout(chunk%tiles(1)%field%energy0) &
  !$fnacc& delete(chunk%tiles(1)%field%density0)
end subroutine

! CHECK-LABEL: func.func @_QPtest_fnacc_enter_exit_data_components
! CHECK: fnacc.data_region_enter
! CHECK: fnacc.copyin
! CHECK: fnacc.create
! CHECK: fnacc.present
! CHECK: fnacc.update_device
! CHECK: fnacc.update_host
! CHECK: fnacc.release
! CHECK: fnacc.copyout
! CHECK: fnacc.delete
! CHECK: fnacc.data_region_exit

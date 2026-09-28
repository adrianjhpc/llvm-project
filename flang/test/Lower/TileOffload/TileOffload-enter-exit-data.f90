! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine test_tileoff_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$tileoff enter data copyin(a) create(c)
  !$tileoff enter data copyin(a, b)
  !$tileoff exit data copyout(a, b) delete(a, b)
  !$tileoff exit data copyout(a, c) delete(a, c)
end subroutine

! CHECK: TileOffload.data_region_enter
! CHECK: TileOffload.copyin
! CHECK: TileOffload.create
! CHECK: TileOffload.data_region_enter
! CHECK: TileOffload.copyin
! CHECK: TileOffload.copyin
! CHECK: TileOffload.copyout
! CHECK: TileOffload.copyout
! CHECK: TileOffload.delete
! CHECK: TileOffload.delete
! CHECK: TileOffload.data_region_exit
! CHECK: TileOffload.copyout
! CHECK: TileOffload.copyout
! CHECK: TileOffload.delete
! CHECK: TileOffload.delete
! CHECK: TileOffload.data_region_exit

subroutine test_tileoff_enter_exit_data_components(chunk)
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

  !$tileoff enter data &
  !$tileoff& copyin(chunk%tiles(1)%field%density0) &
  !$tileoff& create(chunk%tiles(1)%field%energy0)
  !$tileoff present(chunk%tiles(1)%field%density0)
  !$tileoff update device(chunk%tiles(1)%field%density0)
  !$tileoff update host(chunk%tiles(1)%field%density0)
  !$tileoff release(chunk%tiles(1)%field%density0)
  !$tileoff exit data &
  !$tileoff& copyout(chunk%tiles(1)%field%energy0) &
  !$tileoff& delete(chunk%tiles(1)%field%density0)
end subroutine

! CHECK-LABEL: func.func @_QPtest_tileoff_enter_exit_data_components
! CHECK: TileOffload.data_region_enter
! CHECK: TileOffload.copyin
! CHECK: TileOffload.create
! CHECK: TileOffload.present
! CHECK: TileOffload.update_device
! CHECK: TileOffload.update_host
! CHECK: TileOffload.release
! CHECK: TileOffload.copyout
! CHECK: TileOffload.delete
! CHECK: TileOffload.data_region_exit

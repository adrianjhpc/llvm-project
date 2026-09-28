! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_tileoff_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$tileoff enter data copyin(a, b) create(c)
  !$tileoff exit data copyout(c) delete(a, b, c)
end subroutine

subroutine test_tileoff_component_data(chunk)
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

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadEnterDataDirective
! CHECK: TileOffloadCopyinClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: TileOffloadCreateClause
! CHECK: Name = 'c'

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadExitDataDirective
! CHECK: TileOffloadCopyoutClause
! CHECK: Name = 'c'
! CHECK: TileOffloadDeleteClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: Name = 'c'

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadEnterDataDirective
! CHECK: TileOffloadCopyinClause
! CHECK: Name = 'density0'
! CHECK: TileOffloadCreateClause
! CHECK: Name = 'energy0'
! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadPresentDirective
! CHECK: Name = 'density0'
! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadUpdateDeviceDirective
! CHECK: Name = 'density0'
! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadUpdateHostDirective
! CHECK: Name = 'density0'
! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadReleaseDirective
! CHECK: Name = 'density0'
! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadExitDataDirective
! CHECK: TileOffloadCopyoutClause
! CHECK: Name = 'energy0'
! CHECK: TileOffloadDeleteClause
! CHECK: Name = 'density0'

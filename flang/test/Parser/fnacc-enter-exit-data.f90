! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fnacc_enter_exit_data(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc exit data copyout(c) delete(a, b, c)
end subroutine

subroutine test_fnacc_component_data(chunk)
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

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCEnterDataDirective
! CHECK: FnACCCopyinClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: FnACCCreateClause
! CHECK: Name = 'c'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCExitDataDirective
! CHECK: FnACCCopyoutClause
! CHECK: Name = 'c'
! CHECK: FnACCDeleteClause
! CHECK: Name = 'a'
! CHECK: Name = 'b'
! CHECK: Name = 'c'

! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCEnterDataDirective
! CHECK: FnACCCopyinClause
! CHECK: Name = 'density0'
! CHECK: FnACCCreateClause
! CHECK: Name = 'energy0'
! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCPresentDirective
! CHECK: Name = 'density0'
! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCUpdateDeviceDirective
! CHECK: Name = 'density0'
! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCUpdateHostDirective
! CHECK: Name = 'density0'
! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCReleaseDirective
! CHECK: Name = 'density0'
! CHECK: FnACCStandaloneConstruct
! CHECK: FnACCExitDataDirective
! CHECK: FnACCCopyoutClause
! CHECK: Name = 'energy0'
! CHECK: FnACCDeleteClause
! CHECK: Name = 'density0'

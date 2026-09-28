! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_tileoff_data_directives(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$tileoff parallel tile(128) no_copyback pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$tileoff update host(c)
  !$tileoff update device(a)
  !$tileoff release(a, c)
  !$tileoff release all
end subroutine

! CHECK: TileOffloadConstruct
! CHECK: TileOffloadParallelDirective
! CHECK: TileOffloadTileClause
! CHECK: TileOffloadNoCopybackClause
! CHECK: TileOffloadPackClause

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadUpdateHostDirective
! CHECK: Name = 'c'

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadUpdateDeviceDirective
! CHECK: Name = 'a'

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadReleaseDirective
! CHECK: Name = 'a'
! CHECK: Name = 'c'

! CHECK: TileOffloadStandaloneConstruct
! CHECK: TileOffloadReleaseAllDirective


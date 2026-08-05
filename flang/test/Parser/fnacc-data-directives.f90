! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s

subroutine test_fngpu_data_directives(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fngpu parallel tile(128) pack(a:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$fngpu update host(c)
  !$fngpu update device(a)
  !$fngpu release(a, c)
  !$fngpu release all
end subroutine

! CHECK: FnGPUConstruct
! CHECK: FnGPUParallelDirective
! CHECK: FnGPUTileClause
! CHECK: FnGPUPackClause

! CHECK: FnGPUStandaloneConstruct
! CHECK: FnGPUUpdateHostDirective
! CHECK: Name = 'c'

! CHECK: FnGPUStandaloneConstruct
! CHECK: FnGPUUpdateDeviceDirective
! CHECK: Name = 'a'

! CHECK: FnGPUStandaloneConstruct
! CHECK: FnGPUReleaseDirective
! CHECK: Name = 'a'
! CHECK: Name = 'c'

! CHECK: FnGPUStandaloneConstruct
! CHECK: FnGPUReleaseAllDirective


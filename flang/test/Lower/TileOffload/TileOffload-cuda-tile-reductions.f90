! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-triton="backend=cuda-tile allow-backend-fallback=false ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null
! RUN: FileCheck %s --check-prefix=TILE < %t.ttir.cuda-tile
! RUN: FileCheck %s --check-prefix=JSON < %t.json
! RUN: FileCheck %s --check-prefix=EMPTY --allow-empty < %t.ttir
subroutine tile_sum(n,a,s)
integer :: n,i
real :: a(n),s
!$tileoff parallel tile(16) reduction(+:s)
do i=1,n
 s=s+a(i)
enddo
end subroutine
subroutine tile_dot(n,a,b,s)
integer :: n,i
real(8) :: a(n),b(n),s
!$tileoff parallel tile(16) reduction(+:s)
do i=1,n
 s=s+a(i)*b(i)
enddo
end subroutine
! TILE: cuda_tile.module
! TILE: entry @
! TILE: load_ptr_tko weak
! TILE: reduce {{.*}} dim=0 identities=[0.0 : f32]
! TILE: entry @
! TILE: reduce {{.*}} dim=0 identities=[0.0 : f32]
! TILE: entry @
! TILE: mulf %r0_value, %r1_value
! TILE: reduce {{.*}} dim=0 identities=[0.0 : f64]
! TILE: entry @
! TILE: reduce {{.*}} dim=0 identities=[0.0 : f64]
! JSON: "selected_backend": "cuda-tile"
! JSON: "kind": "reduction_sum1d"
! JSON: "threads_per_cta": 1
! JSON: "private_pointer_args": 0
! JSON: "kind": "reduction_stage1d"
! JSON: "threads_per_cta": 1
! JSON: "private_pointer_args": 0
! JSON: "kind": "reduction_dot1d"
! JSON: "threads_per_cta": 1
! JSON: "private_pointer_args": 0
! JSON: "kind": "reduction_stage1d"
! JSON: "threads_per_cta": 1
! JSON: "private_pointer_args": 0
! EMPTY-NOT: tt.func

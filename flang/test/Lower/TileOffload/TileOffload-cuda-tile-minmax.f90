! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --TileOffload-lower-to-triton="backend=cuda-tile allow-backend-fallback=false ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null
! RUN: FileCheck %s --check-prefix=TILE < %t.ttir.cuda-tile
! RUN: FileCheck %s --check-prefix=JSON < %t.json
! RUN: FileCheck %s --check-prefix=EMPTY --allow-empty < %t.ttir
subroutine tile_min(n,a,s)
integer :: n,i
real(4) :: a(n),s
!$tileoff parallel tile(16) reduction(min:s)
do i=1,n
 s=min(s,a(i))
enddo
end subroutine
subroutine tile_max(n,a,s)
integer :: n,i
real(8) :: a(n),s
!$tileoff parallel tile(16) reduction(max:s)
do i=1,n
 s=max(s,a(i))
enddo
end subroutine
! TILE: cuda_tile.module
! TILE: entry @
! TILE: %identity = constant <f32: 0x7F800000>
! TILE: load_ptr_tko weak {{.*}}, %mask, %identity token=
! TILE: reduce {{.*}} dim=0 identities=[0x7F800000 : f32]
! TILE: minf %lhs, %rhs propagate_nan
! TILE: entry @
! TILE: %identity = constant <f32: 0x7F800000>
! TILE: load_ptr_tko weak {{.*}}, %mask, %identity token=
! TILE: reduce {{.*}} dim=0 identities=[0x7F800000 : f32]
! TILE: minf %lhs, %rhs propagate_nan
! TILE: entry @
! TILE: %identity = constant <f64: 0xFFF0000000000000>
! TILE: load_ptr_tko weak {{.*}}, %mask, %identity token=
! TILE: reduce {{.*}} dim=0 identities=[0xFFF0000000000000 : f64]
! TILE: maxf %lhs, %rhs propagate_nan
! TILE: entry @
! TILE: %identity = constant <f64: 0xFFF0000000000000>
! TILE: load_ptr_tko weak {{.*}}, %mask, %identity token=
! TILE: reduce {{.*}} dim=0 identities=[0xFFF0000000000000 : f64]
! TILE: maxf %lhs, %rhs propagate_nan
! JSON: "selected_backend": "cuda-tile"
! JSON: "kind": "reduction_min1d"
! JSON: "kind": "reduction_stage1d"
! JSON: "kind": "reduction_max1d"
! JSON: "kind": "reduction_stage1d"
! EMPTY-NOT: tt.func

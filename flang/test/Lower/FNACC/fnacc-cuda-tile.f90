! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-lower-to-triton="backend=cuda-tile ttir-output=%t.ttir json-output=%t.json" %t.fir -o /dev/null 2>&1 | FileCheck %s --check-prefix=FALLBACK
! RUN: FileCheck %s --check-prefix=TILE < %t.ttir.cuda-tile
! RUN: FileCheck %s --check-prefix=TRITON < %t.ttir
! RUN: FileCheck %s --check-prefix=JSON < %t.json
! RUN: not fir-opt --fnacc-lower-to-triton="backend=cuda-tile allow-backend-fallback=false ttir-output=%t.fail.ttir json-output=%t.fail.json" %t.fir -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRICT
subroutine add(n,a,b,c)
integer :: n,i
real :: a(n),b(n),c(n)
!$fnacc parallel tile(16)
do i=1,n
 c(i)=a(i)+b(i)
enddo
end subroutine
subroutine total(n,a,s)
integer :: n,i
real :: a(n),s
s=0
!$fnacc parallel tile(16) reduction(+:s)
do i=1,n
 s=s+a(i)
enddo
end subroutine
! FALLBACK: falling back to 'triton'
! STRICT: error: FNACC backend selection failed:
! TILE: cuda_tile.module
! TILE: entry @fnacc_kernel_
! TILE: load_ptr_tko
! TILE: addf
! TILE: store_ptr_tko
! TILE-NOT: tt.func
! TRITON: tt.func @fnacc_kernel_
! TRITON: tt.reduce
! JSON: "selected_backend": "mixed"
! JSON: "backend": "cuda-tile"
! JSON: "device_image_kind": "cubin"
! JSON: "threads_per_cta": 1
! JSON: "private_pointer_args": 0
! JSON: "backend": "triton"
! JSON: "private_pointer_args": 2

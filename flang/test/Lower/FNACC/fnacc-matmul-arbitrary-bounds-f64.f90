! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 f64-matmul-strategy=dot ttir-output=%t.2.dot.ttir json-output=%t.2.dot.json" %t.fir -o %t.2.dot.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.2.dot.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.2.dot.json
! RUN: FileCheck %s --check-prefix=V2 --input-file=%t.2.dot.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 f64-matmul-strategy=fma ttir-output=%t.2.fma.ttir json-output=%t.2.fma.json" %t.fir -o %t.2.fma.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.2.fma.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.2.fma.json
! RUN: FileCheck %s --check-prefix=V2 --input-file=%t.2.fma.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=2 f64-matmul-strategy=reduce ttir-output=%t.2.reduce.ttir json-output=%t.2.reduce.json" %t.fir -o %t.2.reduce.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.2.reduce.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.2.reduce.json
! RUN: FileCheck %s --check-prefix=V2 --input-file=%t.2.reduce.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 f64-matmul-strategy=dot ttir-output=%t.3.dot.ttir json-output=%t.3.dot.json" %t.fir -o %t.3.dot.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.3.dot.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.3.dot.json
! RUN: FileCheck %s --check-prefix=V3 --input-file=%t.3.dot.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 f64-matmul-strategy=fma ttir-output=%t.3.fma.ttir json-output=%t.3.fma.json" %t.fir -o %t.3.fma.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.3.fma.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.3.fma.json
! RUN: FileCheck %s --check-prefix=V3 --input-file=%t.3.fma.fir
! RUN: fir-opt --fnacc-pipeline="launch-abi=3 f64-matmul-strategy=reduce ttir-output=%t.3.reduce.ttir json-output=%t.3.reduce.json" %t.fir -o %t.3.reduce.fir
! RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t.3.reduce.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.3.reduce.json
! RUN: FileCheck %s --check-prefix=V3 --input-file=%t.3.reduce.fir

subroutine matmul_bounds_f64(ilo,ihi,jlo,jhi,klo,khi,al,au,bl,bu,cl,cu,a,b,c)
  integer :: ilo,ihi,jlo,jhi,klo,khi,al(2),au(2),bl(2),bu(2),cl(2),cu(2)
  real(8) :: a(al(1):au(1),al(2):au(2))
  real(8) :: b(bl(1):bu(1),bl(2):bu(2))
  real(8) :: c(cl(1):cu(1),cl(2):cu(2)), acc
  integer :: i,j,p
  !$fnacc parallel tile(16,16,8)
  do j=jlo,jhi
    do i=ilo,ihi
      acc=0.0_8
      do p=klo,khi
        acc=acc+a(i,p)*b(p,j)
      enddo
      c(i,j)=acc
    enddo
  enddo
end subroutine

! V2: call @__fnacc_begin_launch_v2
! V2-COUNT-3: call @__fnacc_bind_array_v2
! V2: call @__fnacc_commit_launch_v2
! V3: call @__fnacc_launch_v3
! DEVICE-LABEL: tt.func @fnacc_kernel_0
! DEVICE-SAME: %lx: i32, %ly: i32, %lz: i32
! DEVICE-SAME: %a_l0: i32, %a_l1: i32, %a_s0: i32, %a_s1: i32
! DEVICE-SAME: %b_l0: i32, %b_l1: i32, %b_s0: i32, %b_s1: i32
! DEVICE-SAME: %c_l0: i32, %c_l1: i32, %c_s0: i32, %c_s1: i32
! DEVICE: arith.extsi
! DEVICE: %offs_m_e_c = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
! DEVICE: %offs_n_e_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
! DEVICE: %c_addr_row64 = arith.extsi %offs_m_b_c : tensor<16x16xi32> to tensor<16x16xi64>
! DEVICE: %c_addr_col64 = arith.extsi %offs_n_b_c : tensor<16x16xi32> to tensor<16x16xi64>
! DEVICE: %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<16x16x!tt.ptr<f64>>, tensor<16x16xi64>
! DEVICE: tt.store
! JSON: "kind": "matmul2d"
! JSON: "launch_abi_version": 2
! JSON: "role": "loop_lower_z"
! JSON: "role": "array_lower_bound"
! JSON: "role": "array_stride"

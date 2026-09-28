! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --TileOffload-pipeline="launch-abi=2 ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine pack_unpack(x_min, x_max, y_min, y_max, x_inc, &
                                   y_inc, depth, buffer_offset, field, &
                                   left_snd_buffer, left_rcv_buffer)
  implicit none
  integer :: x_min, x_max, y_min, y_max, x_inc, y_inc
  integer :: depth, buffer_offset
  real(8) :: field(:, :)
  real(8) :: left_snd_buffer(:), left_rcv_buffer(:)
  integer :: index, j, k

  !$tileoff parallel tile(16, 16)
  do k = y_min-depth, y_max+y_inc+depth
    do j = 1, depth
      index = buffer_offset + j + (k+depth-1)*depth
      left_snd_buffer(index) = field(x_min+x_inc-1+j, k)
    end do
  end do

  !$tileoff parallel tile(16, 16)
  do k = y_min-depth, y_max+y_inc+depth
    do j = 1, depth
      index = buffer_offset + j + (k+depth-1)*depth
      field(x_min-j, k) = left_rcv_buffer(index)
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__tileoff_begin_launch_v2
! HOST-DAG: func.func private @__tileoff_bind_array_v2
! HOST-DAG: func.func private @__tileoff_bind_scalar_i32_v2
! HOST-DAG: func.func private @__tileoff_commit_launch_v2
! HOST-LABEL: func.func @_QPpack_unpack
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-2: call @__tileoff_bind_array_v2
! HOST-COUNT-4: call @__tileoff_bind_scalar_i32_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST: call @__tileoff_begin_launch_v2
! HOST-COUNT-2: call @__tileoff_bind_array_v2
! HOST-COUNT-3: call @__tileoff_bind_scalar_i32_v2
! HOST: call @__tileoff_commit_launch_v2
! HOST-NOT: TileOffload.launch

! TTIR-LABEL: tt.func @tileoff_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %index0: i32
! TTIR-SAME: %index3: i32
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR: arith.muli
! TTIR: tt.load
! TTIR: tt.store

! TTIR-LABEL: tt.func @tileoff_kernel_1(
! TTIR-SAME: %array0: !tt.ptr<f64>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %index0: i32
! TTIR-SAME: %index2: i32
! TTIR-SAME: %extent_x: i32
! TTIR-SAME: %extent_y: i32
! TTIR: arith.muli
! TTIR: tt.load
! TTIR: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 2
! JSON: "scalar_count": 4
! JSON: "output_count": 1
! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 2
! JSON: "scalar_count": 3
! JSON: "output_count": 1

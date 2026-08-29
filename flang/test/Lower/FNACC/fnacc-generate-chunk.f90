! RUN: %flang_fc1 -emit-fir %s -o %t.fir
! RUN: fir-opt \
! RUN:   --fnacc-pipeline="ttir-output=%t.ttir json-output=%t.json" \
! RUN:   %t.fir -o %t.host.fir
! RUN: FileCheck %s --check-prefix=HOST --input-file=%t.host.fir
! RUN: FileCheck %s --check-prefix=TTIR --input-file=%t.ttir
! RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
! RUN: %python -m json.tool %t.json > /dev/null

subroutine generate_chunk( &
    x_min, x_max, y_min, y_max, s, vertexx, vertexy, cellx, celly, &
    density0, en, xvel0, yvel0, sd, se, &
    xvel, yvel, xmin, xmax, ymin, &
    ymax, sr, sg, g_rect, g_circ, g_point)
  implicit none
  integer :: x_min, x_max, y_min, y_max, s
  integer :: g_rect, g_circ, g_point
  real(8) :: vertexx(x_min-2:x_max+3), vertexy(y_min-2:y_max+3)
  real(8) :: cellx(x_min-2:x_max+2), celly(y_min-2:y_max+2)
  real(8) :: density0(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: en(x_min-2:x_max+2, y_min-2:y_max+2)
  real(8) :: xvel0(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: yvel0(x_min-2:x_max+3, y_min-2:y_max+3)
  real(8) :: sd(:), se(:)
  real(8) :: xvel(:), yvel(:)
  real(8) :: xmin(:), xmax(:)
  real(8) :: ymin(:), ymax(:), sr(:)
  integer :: sg(:)
  real(8) :: radius, x_cent, y_cent
  integer :: j, k, jt, kt

  x_cent = xmin(s)
  y_cent = ymin(s)

  !$fnacc parallel tile(16, 16)
  do k = y_min-2, y_max+2
    do j = x_min-2, x_max+2
      if (sg(s) == g_rect) then
        if (vertexx(j+1) >= xmin(s) .and. &
            vertexx(j) < xmax(s)) then
          if (vertexy(k+1) >= ymin(s) .and. &
              vertexy(k) < ymax(s)) then
            en(j, k) = se(s)
            density0(j, k) = sd(s)
            do kt = k, k+1
              do jt = j, j+1
                xvel0(jt, kt) = xvel(s)
                yvel0(jt, kt) = yvel(s)
              end do
            end do
          end if
        end if
      else if (sg(s) == g_circ) then
        radius = sqrt((cellx(j)-x_cent)*(cellx(j)-x_cent) + &
                      (celly(k)-y_cent)*(celly(k)-y_cent))
        if (radius <= sr(s)) then
          en(j, k) = se(s)
          density0(j, k) = sd(s)
          do kt = k, k+1
            do jt = j, j+1
              xvel0(jt, kt) = xvel(s)
              yvel0(jt, kt) = yvel(s)
            end do
          end do
        end if
      else if (sg(s) == g_point) then
        if (vertexx(j) == x_cent .and. vertexy(k) == y_cent) then
          en(j, k) = se(s)
          density0(j, k) = sd(s)
          do kt = k, k+1
            do jt = j, j+1
              xvel0(jt, kt) = xvel(s)
              yvel0(jt, kt) = yvel(s)
            end do
          end do
        end if
      end if
    end do
  end do
end subroutine

! HOST-DAG: func.func private @__fnacc_begin_launch_v2
! HOST-DAG: func.func private @__fnacc_bind_array_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_f64_v2
! HOST-DAG: func.func private @__fnacc_bind_scalar_i32_v2
! HOST-DAG: func.func private @__fnacc_commit_launch_v2
! HOST-LABEL: func.func @_QPgenerate_chunk
! HOST: call @__fnacc_begin_launch_v2
! HOST-COUNT-18: call @__fnacc_bind_array_v2
! HOST-COUNT-2: call @__fnacc_bind_scalar_f64_v2
! HOST-COUNT-4: call @__fnacc_bind_scalar_i32_v2
! HOST: call @__fnacc_commit_launch_v2
! HOST-NOT: fnacc.launch

! TTIR-LABEL: tt.func @fnacc_kernel_0(
! TTIR-SAME: %array0: !tt.ptr<i32>
! TTIR-SAME: %array1: !tt.ptr<f64>
! TTIR-SAME: %scalar0: f64
! TTIR-SAME: %scalar1: f64
! TTIR-SAME: %index0: i32
! TTIR-SAME: %index3: i32
! TTIR: arith.cmpi eq
! TTIR: arith.cmpf
! TTIR: math.sqrt
! TTIR: arith.xori
! TTIR-COUNT-30: tt.store

! JSON: "kind": "stencil2d"
! JSON: "rank": 2
! JSON: "launch_abi_version": 2
! JSON: "array_count": 18
! JSON: "scalar_count": 6
! JSON: "output_count": 30

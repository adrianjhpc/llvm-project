// RUN: fir-opt \
// RUN:   --TileOffload-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @data_ops(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %x: !fir.ref<f32>,
      %y: !fir.ref<f32>) {
    TileOffload.update_host %a : !fir.ref<!fir.array<?xf32>>
    TileOffload.update_device %a : !fir.ref<!fir.array<?xf32>>
    TileOffload.present %a : !fir.ref<!fir.array<?xf32>>
    TileOffload.release %a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
    TileOffload.release_all
    TileOffload.wait

    TileOffload.data_region_enter
    TileOffload.copyin %x : !fir.ref<f32>
    TileOffload.create %y : !fir.ref<f32>
    TileOffload.copyout %y : !fir.ref<f32>
    TileOffload.delete %x : !fir.ref<f32>
    TileOffload.data_region_exit

    return
  }
}

// CHECK-DAG: func.func private @__tileoff_update_host(!fir.ref<i8>)
// CHECK-DAG: func.func private @__tileoff_update_device(!fir.ref<i8>)
// CHECK-DAG: func.func private @__tileoff_present(!fir.ref<i8>)
// CHECK-DAG: func.func private @__tileoff_release(!fir.ref<i8>)
// CHECK-DAG: func.func private @__tileoff_release_all()
// CHECK-DAG: func.func private @__tileoff_wait()
// CHECK-DAG: func.func private @__tileoff_enter_data_region()
// CHECK-DAG: func.func private @__tileoff_data_copyin_bytes
// CHECK-DAG: func.func private @__tileoff_data_create_bytes
// CHECK-DAG: func.func private @__tileoff_data_copyout_bytes
// CHECK-DAG: func.func private @__tileoff_data_delete
// CHECK-DAG: func.func private @__tileoff_exit_data_region()

// CHECK-LABEL: func.func @data_ops
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__tileoff_update_host
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__tileoff_update_device
// CHECK: call @__tileoff_present
// CHECK: call @__tileoff_release
// CHECK: call @__tileoff_release
// CHECK: call @__tileoff_release_all
// CHECK: call @__tileoff_wait
// CHECK: call @__tileoff_enter_data_region
// CHECK: call @__tileoff_data_copyin_bytes
// CHECK: call @__tileoff_data_create_bytes
// CHECK: call @__tileoff_data_copyout_bytes
// CHECK: call @__tileoff_data_delete
// CHECK: call @__tileoff_exit_data_region

// CHECK-NOT: TileOffload.update_host
// CHECK-NOT: TileOffload.update_device
// CHECK-NOT: TileOffload.present
// CHECK-NOT: TileOffload.release
// CHECK-NOT: TileOffload.release_all
// CHECK-NOT: TileOffload.wait
// CHECK-NOT: TileOffload.data_region_enter
// CHECK-NOT: TileOffload.data_region_exit
// CHECK-NOT: TileOffload.copyin
// CHECK-NOT: TileOffload.create
// CHECK-NOT: TileOffload.copyout
// CHECK-NOT: TileOffload.delete


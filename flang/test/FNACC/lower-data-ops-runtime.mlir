// RUN: fir-opt \
// RUN:   --fnacc-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @data_ops(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>,
      %x: !fir.ref<f32>,
      %y: !fir.ref<f32>) {
    fnacc.update_host %a : !fir.ref<!fir.array<?xf32>>
    fnacc.update_device %a : !fir.ref<!fir.array<?xf32>>
    fnacc.present %a : !fir.ref<!fir.array<?xf32>>
    fnacc.release %a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
    fnacc.release_all
    fnacc.wait

    fnacc.data_region_enter
    fnacc.copyin %x : !fir.ref<f32>
    fnacc.create %y : !fir.ref<f32>
    fnacc.copyout %y : !fir.ref<f32>
    fnacc.delete %x : !fir.ref<f32>
    fnacc.data_region_exit

    return
  }
}

// CHECK-DAG: func.func private @__fnacc_update_host(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_update_device(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_present(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_release(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_release_all()
// CHECK-DAG: func.func private @__fnacc_wait()
// CHECK-DAG: func.func private @__fnacc_enter_data_region()
// CHECK-DAG: func.func private @__fnacc_data_copyin_bytes
// CHECK-DAG: func.func private @__fnacc_data_create_bytes
// CHECK-DAG: func.func private @__fnacc_data_copyout_bytes
// CHECK-DAG: func.func private @__fnacc_data_delete
// CHECK-DAG: func.func private @__fnacc_exit_data_region()

// CHECK-LABEL: func.func @data_ops
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fnacc_update_host
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fnacc_update_device
// CHECK: call @__fnacc_present
// CHECK: call @__fnacc_release
// CHECK: call @__fnacc_release
// CHECK: call @__fnacc_release_all
// CHECK: call @__fnacc_wait
// CHECK: call @__fnacc_enter_data_region
// CHECK: call @__fnacc_data_copyin_bytes
// CHECK: call @__fnacc_data_create_bytes
// CHECK: call @__fnacc_data_copyout_bytes
// CHECK: call @__fnacc_data_delete
// CHECK: call @__fnacc_exit_data_region

// CHECK-NOT: fnacc.update_host
// CHECK-NOT: fnacc.update_device
// CHECK-NOT: fnacc.present
// CHECK-NOT: fnacc.release
// CHECK-NOT: fnacc.release_all
// CHECK-NOT: fnacc.wait
// CHECK-NOT: fnacc.data_region_enter
// CHECK-NOT: fnacc.data_region_exit
// CHECK-NOT: fnacc.copyin
// CHECK-NOT: fnacc.create
// CHECK-NOT: fnacc.copyout
// CHECK-NOT: fnacc.delete


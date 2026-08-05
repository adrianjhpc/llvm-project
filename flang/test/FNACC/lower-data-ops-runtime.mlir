// RUN: fir-opt \
// RUN:   --fngpu-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @data_ops(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>) {
    fngpu.update_host %a : !fir.ref<!fir.array<?xf32>>
    fngpu.update_device %a : !fir.ref<!fir.array<?xf32>>
    fngpu.release %a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
    fngpu.release_all

    return
  }
}

// CHECK-DAG: func.func private @__fngpu_update_host(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fngpu_update_device(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fngpu_release(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fngpu_release_all()

// CHECK-LABEL: func.func @data_ops
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fngpu_update_host
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fngpu_update_device
// CHECK: call @__fngpu_release
// CHECK: call @__fngpu_release
// CHECK: call @__fngpu_release_all

// CHECK-NOT: fngpu.update_host
// CHECK-NOT: fngpu.update_device
// CHECK-NOT: fngpu.release
// CHECK-NOT: fngpu.release_all


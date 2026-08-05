// RUN: fir-opt \
// RUN:   --fnacc-lower-to-runtime \
// RUN:   %s | FileCheck %s

module {
  func.func @data_ops(
      %a: !fir.ref<!fir.array<?xf32>>,
      %b: !fir.ref<!fir.array<?xf32>>) {
    fnacc.update_host %a : !fir.ref<!fir.array<?xf32>>
    fnacc.update_device %a : !fir.ref<!fir.array<?xf32>>
    fnacc.release %a, %b : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>
    fnacc.release_all

    return
  }
}

// CHECK-DAG: func.func private @__fnacc_update_host(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_update_device(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_release(!fir.ref<i8>)
// CHECK-DAG: func.func private @__fnacc_release_all()

// CHECK-LABEL: func.func @data_ops
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fnacc_update_host
// CHECK: fir.convert {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> !fir.ref<i8>
// CHECK: call @__fnacc_update_device
// CHECK: call @__fnacc_release
// CHECK: call @__fnacc_release
// CHECK: call @__fnacc_release_all

// CHECK-NOT: fnacc.update_host
// CHECK-NOT: fnacc.update_device
// CHECK-NOT: fnacc.release
// CHECK-NOT: fnacc.release_all


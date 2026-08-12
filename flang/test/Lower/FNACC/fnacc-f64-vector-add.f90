!RUN : % flang_fc1 - emit - fir % s - o % t.fir !RUN : fir - opt-- fnacc -
    pipeline = "ttir-output=%t.ttir json-output=%t.json" % t.fir -
    o % t.host.fir !RUN : FileCheck % s-- check - prefix = HOST-- input -
    file = % t.host.fir !RUN : FileCheck % s-- check - prefix = TTIR-- input -
    file = % t.ttir !RUN : FileCheck % s-- check - prefix = JSON-- input -
    file = % t.json !RUN : python3 - m json.tool % t.json > / dev /
        null

            subroutine
            vector_add_f64(n, a, b, c) integer::n real(8)::a(n),
    b(n),
    c(n) integer::i

    !$fnacc parallel tile(128) do i = 1,
    n c(i) = a(i) +
    b(i) end do end subroutine

    !HOST : call @__fnacc_launch_f64_v1

            !TTIR : tt.func @fnacc_kernel_0 !TTIR
    -
    SAME : !tt.ptr<f64> !TTIR : tensor<128xf64> !TTIR : arith.addf !TTIR
    : tt.store

      !JSON : "type" : "ptr<f64>"

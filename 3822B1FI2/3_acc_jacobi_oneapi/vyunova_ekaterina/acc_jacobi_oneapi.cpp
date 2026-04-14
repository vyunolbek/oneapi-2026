#include "acc_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {

    const int n = static_cast<int>(b.size());

    std::vector<float> x_old(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    sycl::buffer<float> a_buf(a.data(), a.size());
    sycl::buffer<float> b_buf(b.data(), b.size());
    sycl::buffer<float> x_old_buf(x_old.data(), x_old.size());
    sycl::buffer<float> x_new_buf(x_new.data(), x_new.size());

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc   = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc   = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto old_acc = x_old_buf.get_access<sycl::access::mode::read>(cgh);
            auto new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                int i = static_cast<int>(id[0]);
                float sum = 0.0f;

                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * old_acc[j];
                    }
                }

                new_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
            });
        }).wait();

        bool converged = true;
        {
            auto old_acc = x_old_buf.get_host_access();
            auto new_acc = x_new_buf.get_host_access();

            for (int i = 0; i < n; ++i) {
                if (std::fabs(new_acc[i] - old_acc[i]) >= accuracy) {
                    converged = false;
                }
                old_acc[i] = new_acc[i];
            }
        }

        if (converged) {
            break;
        }
    }

    std::vector<float> result(n);
    {
        auto new_acc = x_new_buf.get_host_access();
        for (int i = 0; i < n; ++i) {
            result[i] = new_acc[i];
        }
    }
    return result;
}

#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    size_t N = b.size();
    sycl::queue q(device);

    std::vector<float> x(N, 0.0f);       
    std::vector<float> x_new(N, 0.0f);   
    std::vector<float> diff(N, 0.0f);    

    sycl::buffer<float, 1> bufA(a.data(), sycl::range<1>(N * N));
    sycl::buffer<float, 1> bufB(b.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> bufX(x.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> bufXNew(x_new.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> bufDiff(diff.data(), sycl::range<1>(N));

    float residual = accuracy + 1;
    int iter = 0;

    while (residual > accuracy && iter < ITERATIONS) {
        ++iter;

        q.submit([&](sycl::handler& h) {
            auto a_acc = bufA.get_access<sycl::access::mode::read>(h);
            auto b_acc = bufB.get_access<sycl::access::mode::read>(h);
            auto x_acc = bufX.get_access<sycl::access::mode::read>(h);
            auto x_new_acc = bufXNew.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                float sum = 0.0f;
                float diag = 0.0f;
                int row = i[0];
                for (int col = 0; col < N; ++col) {
                    float val = a_acc[row * N + col];
                    if (col == row) {
                        diag = val;
                    } else {
                        sum += val * x_acc[col];
                    }
                }
                x_new_acc[row] = (b_acc[row] - sum) / diag;
            });
        });
        
        q.submit([&](sycl::handler& h) {
            auto x_acc = bufX.get_access<sycl::access::mode::read>(h);
            auto x_new_acc = bufXNew.get_access<sycl::access::mode::read>(h);
            auto diff_acc = bufDiff.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                diff_acc[i[0]] = std::abs(x_new_acc[i[0]] - x_acc[i[0]]);
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto x_acc = bufX.get_access<sycl::access::mode::write>(h);
            auto x_new_acc = bufXNew.get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                x_acc[i[0]] = x_new_acc[i[0]];
            });
        }).wait();

        auto diff_host = diff.data();
        residual = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            residual = std::max(residual, diff_host[i]);
        }
    }

    {
        auto host_x_acc = bufX.get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < N; ++i) {
            x[i] = host_x_acc[i];
        }
    }

    return x;

}

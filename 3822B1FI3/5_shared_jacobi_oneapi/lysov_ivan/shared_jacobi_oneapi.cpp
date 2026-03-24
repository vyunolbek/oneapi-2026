#include "shared_jacobi_oneapi.h"

#include <algorithm>
#include <cmath>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t n = b.size();

    if (n == 0) {
        return {};
    }

    if (a.size() != n * n) {
        return {};
    }

    if (accuracy < 0.0f) {
        accuracy = 0.0f;
    }

    sycl::queue q(device);

    float* s_a = sycl::malloc_shared<float>(a.size(), q);
    float* s_b = sycl::malloc_shared<float>(b.size(), q);
    float* s_x_old = sycl::malloc_shared<float>(n, q);
    float* s_x_new = sycl::malloc_shared<float>(n, q);
    float* s_diff = sycl::malloc_shared<float>(n, q);

    if (s_a == nullptr || s_b == nullptr || s_x_old == nullptr ||
        s_x_new == nullptr || s_diff == nullptr) {
        if (s_a) sycl::free(s_a, q);
        if (s_b) sycl::free(s_b, q);
        if (s_x_old) sycl::free(s_x_old, q);
        if (s_x_new) sycl::free(s_x_new, q);
        if (s_diff) sycl::free(s_diff, q);
        return {};
    }

    std::copy(a.begin(), a.end(), s_a);
    std::copy(b.begin(), b.end(), s_b);
    std::fill(s_x_old, s_x_old + n, 0.0f);
    std::fill(s_x_new, s_x_new + n, 0.0f);
    std::fill(s_diff, s_diff + n, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            const size_t row_offset = i * n;

            float row_sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    row_sum += s_a[row_offset + j] * s_x_old[j];
                }
            }

            const float diag = s_a[row_offset + i];
            const float new_value = (s_b[i] - row_sum) / diag;

            s_x_new[i] = new_value;
            s_diff[i] = sycl::fabs(new_value - s_x_old[i]);
        }).wait();

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, s_diff[i]);
        }

        if (max_diff < accuracy) {
            converged = true;
            break;
        }

        std::swap(s_x_old, s_x_new);
    }

    const float* source = converged ? s_x_new : s_x_old;
    std::vector<float> result(source, source + n);

    sycl::free(s_a, q);
    sycl::free(s_b, q);
    sycl::free(s_x_old, q);
    sycl::free(s_x_new, q);
    sycl::free(s_diff, q);

    return result;
}
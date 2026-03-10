#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;

    const float dx = (end - start) / static_cast<float>(count);
    const float dy = dx;

    float result = 0.0f;

    try {
        sycl::queue q{device};

        q.submit([&](sycl::handler& h) {
            auto sum_red = sycl::reduction(result, sycl::plus<float>());

            h.parallel_for(
                sycl::range<2>(count, count),
                sum_red,
                [=](sycl::id<2> idx, auto& sum) {
                    const int i = idx[1];
                    const int j = idx[0];

                    const float x_mid = start + (static_cast<float>(i) + 0.5f) * dx;
                    const float y_mid = start + (static_cast<float>(j) + 0.5f) * dy;

                    sum += sycl::sin(x_mid) * sycl::cos(y_mid) * dx * dy;
                });
        }).wait();
    }
    catch (sycl::exception const& e) {
        return 0.0f;
    }

    return result;
}
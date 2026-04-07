#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / static_cast<float>(count);
    const size_t total_cells =
        static_cast<size_t>(count) * static_cast<size_t>(count);

    float sum_value = 0.0f;
    sycl::queue queue(device);

    {
        sycl::buffer<float> sum_buffer(&sum_value, sycl::range<1>(1));

        queue.submit([&](sycl::handler& cgh) {
            auto sum_reduction =
                sycl::reduction(sum_buffer, cgh, sycl::plus<float>());

            cgh.parallel_for(
                sycl::range<1>(total_cells),
                sum_reduction,
                [=](sycl::id<1> index, auto& partial_sum) {
                    const size_t pos = index[0];
                    const int ix = static_cast<int>(pos / static_cast<size_t>(count));
                    const int iy = static_cast<int>(pos % static_cast<size_t>(count));

                    const float x = start + (static_cast<float>(ix) + 0.5f) * step;
                    const float y = start + (static_cast<float>(iy) + 0.5f) * step;

                    partial_sum += sycl::sin(x) * sycl::cos(y);
                });
        });

        queue.wait();
    }

    return sum_value * step * step;
}
#include "dev_jacobi_oneapi.h"
#include <algorithm>

using buftype = sycl::buffer<float>;

std::vector<float> JacobiDevONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, float accuracy,
                                   sycl::device device) {
  size_t size = b.size();
  std::vector<float> prev_res(size, 0.0f);
  std::vector<float> res(size, 0.0f);

  auto stop = [&]() -> bool {
    float norm = 0.0f;
    for (int idx = 0; idx < prev_res.size(); ++idx) {
      float el = res[idx] - prev_res[idx];
      norm += el * el;
    }
    return norm < accuracy * accuracy;
  };

  sycl::queue gpu_queue(device);

  auto alloc = [&gpu_queue](size_t size) -> float * {
    return sycl::malloc_device<float>(size, gpu_queue);
  };

  auto free = [&gpu_queue](void *ptr) -> void { sycl::free(ptr, gpu_queue); };

  float *in_a = alloc(a.size());
  float *in_b = alloc(b.size());
  float *in_prev_res = alloc(prev_res.size());
  float *in_res = alloc(res.size());

  gpu_queue.memcpy(in_a, a.data(), sizeof(float) * a.size()).wait();
  gpu_queue.memcpy(in_b, b.data(), sizeof(float) * b.size()).wait();

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {

    gpu_queue.memcpy(in_prev_res, prev_res.data(), sizeof(float) * size).wait();

    gpu_queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
            int idx = id.get(0);
            float next_res = 0;

            for (int indx = 0; indx < size; indx++) {
              if (idx == indx) {
                next_res += in_b[indx];
              } else {
                next_res -= in_a[idx * size + indx] * in_prev_res[indx];
              }
            }
            next_res /= in_a[idx * size + idx];
            in_res[idx] = next_res;
          });
        })
        .wait();

    gpu_queue.memcpy(res.data(), in_res, sizeof(float) * size).wait();

    if (stop()) {
      break;
    }

    std::swap(prev_res, res);
  }

  free(in_a);
  free(in_b);
  free(in_prev_res);
  free(in_res);

  return res;
}

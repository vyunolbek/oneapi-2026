#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {

    const size_t matrix_elements = a.size();
    const size_t dim = static_cast<size_t>(std::sqrt(matrix_elements));

    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

    Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> mat("matrix", dim, dim);
    Kokkos::View<float*, MemSpace> rhs("rhs", dim);
    Kokkos::View<float*, MemSpace> current("current", dim);
    Kokkos::View<float*, MemSpace> next("next", dim);
    Kokkos::View<float*, MemSpace> previous("previous", dim);

    auto mat_h = Kokkos::create_mirror_view(mat);
    auto rhs_h = Kokkos::create_mirror_view(rhs);

    for (size_t r = 0; r < dim; ++r) {
        rhs_h(r) = b[r];
        for (size_t c = 0; c < dim; ++c) {
            mat_h(r, c) = a[r * dim + c];
        }
    }

    Kokkos::deep_copy(mat, mat_h);
    Kokkos::deep_copy(rhs, rhs_h);

    Kokkos::deep_copy(current, 0.0f);
    Kokkos::deep_copy(next, 0.0f);

    auto compute_step = KOKKOS_LAMBDA(const int row) {
        float accum = 0.0f;
        float diag = mat(row, row);

        for (int col = 0; col < dim; ++col) {
            if (col != row) {
                accum += mat(row, col) * current(col);
            }
        }

        next(row) = (rhs(row) - accum) / diag;
    };

    bool stop = false;

    for (int iteration = 0; iteration < ITERATIONS && !stop; ++iteration) {

        Kokkos::deep_copy(previous, current);

        Kokkos::parallel_for("JacobiKernel", dim, compute_step);
        Kokkos::fence();

        Kokkos::deep_copy(current, next);

        auto host_cur = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, current);
        auto host_prev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, previous);

        stop = true;

        for (size_t i = 0; i < dim; ++i) {
            if (std::fabs(host_cur(i) - host_prev(i)) >= accuracy) {
                stop = false;
                break;
            }
        }
    }

    std::vector<float> solution(dim);

    auto final_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, current);

    for (size_t i = 0; i < dim; ++i) {
        solution[i] = final_host(i);
    }

    return solution;
}

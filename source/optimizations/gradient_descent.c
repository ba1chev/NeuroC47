#include "source/optimizations/gradient_descent.h"

vector_t* calculate_gradient_descent(const gradient_descent_t* gd_optimizer, vector_t* X) {
    if (!gd_optimizer || !X || !X->data) {
        errx(1, "Nullptr detected");
    }

    for (size_t iter = 0; iter < gd_optimizer->count_of_iterations; iter++) {
        vector_t* gradient = calculate_derivative_at(&gd_optimizer->derivative_obj, X);
        avx_256_vector_axpy(X, gradient, -gd_optimizer->learning_rate);
        free_vector(gradient);
        free(gradient);
    }

    return X;
}

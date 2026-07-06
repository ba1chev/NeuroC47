#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "derivatives/derivative.h"
#include "structures/vector/vector.h"

typedef struct gradient_descent_t {
    derivative_t derivative_obj;
    size_t count_of_iterations;
    float learning_rate;
} gradient_descent_t;

vector_t* calculate_gradient_descent(const gradient_descent_t* gd_optimizer, vector_t* X);

#endif
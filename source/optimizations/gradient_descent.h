#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "source/derivatives/derivative.h"
#include "source/structures/vector/vector.h"

typedef struct gradient_descent_t {
    derivative_t derivative_obj;
    size_t count_of_iterations;
    float learning_rate;
} gradient_descent_t;

vector_t* calculate_gradient_descent(const gradient_descent_t* gd_optimizer, vector_t* X);

#endif
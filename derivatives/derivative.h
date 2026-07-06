#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include "functions/function.h"
#include "structures/vector/vector.h"

typedef struct derivative_t {
    function_t function;
    float precision;
} derivative_t;

vector_t* calculate_derivative_at(const derivative_t* derivative_obj, const vector_t* X);

#endif
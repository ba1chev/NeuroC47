#ifndef MSE_H
#define MSE_H

#include "functions/function.h"
#include "structures/vector/vector.h"

float calculate_mse(function_t fn_predicate, const vector_t* X, const vector_t* y);

#endif
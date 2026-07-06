#ifndef MSE_H
#define MSE_H

#include "source/functions/function.h"
#include "source/structures/vector/vector.h"

float calculate_mse(function_t fn_predicate, const vector_t* X, const vector_t* y);

#endif
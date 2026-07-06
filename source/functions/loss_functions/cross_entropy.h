#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "source/structures/vector/vector.h"

float calculate_cross_entropy(const vector_t* predictions, const vector_t* y);
vector_t* cross_entropy_softmax_delta(const vector_t* predictions, const vector_t* y);

#endif

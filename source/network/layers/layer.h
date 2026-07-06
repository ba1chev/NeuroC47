#ifndef LAYER_H
#define LAYER_H

#include "source/functions/function.h"
#include "source/structures/matrix/matrix.h"
#include "source/structures/vector/vector.h"

typedef struct layer_t {
    matrix_t* weights;
    vector_t* biases;
    function_t activation;
} layer_t;

layer_t* init_layer(size_t input_size, size_t output_size, function_t activation);
void free_layer(layer_t* layer);
vector_t* layer_forward(const layer_t* layer, const vector_t* input);

#endif

#ifndef LAYER_H
#define LAYER_H

#include "source/functions/function.h"
#include "source/structures/matrix/matrix.h"
#include "source/structures/vector/vector.h"

typedef struct forward_cache_t forward_cache_t;

typedef struct layer_t {
    matrix_t* weights;
    vector_t* biases;
    function_t activation;
    function_t activation_derivative;
} layer_t;

layer_t* init_layer(size_t input_size, size_t output_size, function_t activation, function_t activation_derivative);
void free_layer(layer_t* layer);
void layer_forward(const layer_t* layer, const vector_t* input, forward_cache_t* cache, size_t layer_index);

#endif

#ifndef NETWORK_H
#define NETWORK_H

#include "network/layers/layer.h"
#include "network/forward_cache.h"
#include "functions/loss_functions/mean_squared_error.h"

typedef struct network_t {
    layer_t** layers;
    size_t count_of_layers;
} network_t;

network_t* init_network(size_t count_of_layers);
void free_network(network_t* network);
void network_add_layer(network_t* network, layer_t* layer);
forward_cache_t* network_forward(const network_t* network, const vector_t* X);
void network_backward(network_t* network, const forward_cache_t* cache, const vector_t* y, float learning_rate);

#endif

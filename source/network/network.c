#include "source/network/network.h"

network_t* init_network(size_t count_of_layers) {
    network_t* network = (network_t*)malloc(sizeof(network_t));
    if (!network) {
        errx(1, "malloc failed");
    }

    network->layers = (layer_t**)malloc(count_of_layers * sizeof(layer_t*));
    if (!network->layers) {
        errx(1, "malloc failed");
    }

    network->count_of_layers = count_of_layers;
    return network;
}

void free_network(network_t* network) {
    if (!network) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < network->count_of_layers; i++) {
        free_layer(network->layers[i]);
    }

    free(network->layers);
    free(network);
}

void network_add_layer(network_t* network, layer_t* layer) {
    if (!network || !layer) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < network->count_of_layers; i++) {
        if (!network->layers[i]) {
            network->layers[i] = layer;
            return;
        }
    }

    errx(1, "Network is full");
}

forward_cache_t* network_forward(const network_t* network, const vector_t* X) {
    if (!network || !X || !X->data) {
        errx(1, "Nullptr detected");
    }

    forward_cache_t* cache = init_forward_cache(network->count_of_layers + 1);
    cache->activations[0] = (vector_t*)malloc(sizeof(vector_t));
    if (!cache->activations[0]) {
        errx(1, "malloc failed");
    }

    init_vector(cache->activations[0]);
    copy_vector(cache->activations[0], X);

    for (size_t i = 0; i < network->count_of_layers; i++) {
        cache->activations[i + 1] = layer_forward(network->layers[i], cache->activations[i]);
    }

    return cache;
}

void network_backward(network_t* network, const forward_cache_t* cache, const vector_t* y, float learning_rate) {
    if (!network || !cache || !y || !y->data) {
        errx(1, "Nullptr detected");
    }

    size_t count_of_layers = network->count_of_layers;
    vector_t* delta = (vector_t*)malloc(sizeof(vector_t));
    if (!delta) {
        errx(1, "malloc failed");
    }
    init_vector(delta);
    copy_vector(delta, cache->activations[count_of_layers]);
    avx_256_vector_sub(delta, y, delta);

    for (size_t layer_index = count_of_layers; layer_index-- > 0;) {
        vector_t* input = cache->activations[layer_index];
        layer_t* layer  = network->layers[layer_index];

        avx_256_vector_axpy(layer->biases, delta, -learning_rate);
        for (size_t i = 0; i < layer->weights->size; i++) {
            avx_256_vector_axpy(layer->weights->data[i], input, -learning_rate * delta->data[i]);
        }

        if (layer_index > 0) {
            vector_t* prev_delta = (vector_t*)malloc(sizeof(vector_t));
            if (!prev_delta) {
                errx(1, "malloc failed");
            }

            init_vector(prev_delta);
            prev_delta->data = (float*)calloc(input->size, sizeof(float));
            if (!prev_delta->data) {
                errx(1, "malloc failed");
            }

            prev_delta->size = input->size;
            prev_delta->capacity = input->size;
            for (size_t j = 0; j < input->size; j++) {
                for (size_t i = 0; i < layer->weights->size; i++) {
                    prev_delta->data[j] += delta->data[i] * layer->weights->data[i]->data[j];
                }
            }

            free_vector(delta);
            free(delta);
            delta = prev_delta;
        }
    }

    free_vector(delta);
    free(delta);
}

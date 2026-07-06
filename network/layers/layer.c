#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "network/layers/layer.h"

layer_t* init_layer(size_t input_size, size_t output_size, function_t activation) {
    layer_t* layer = (layer_t*)malloc(sizeof(layer_t));
    if (!layer) {
        errx(1, "malloc failed");
    }

    float limit = 1.0f / sqrtf((float)input_size);

    layer->weights = (matrix_t*)malloc(sizeof(matrix_t));
    if (!layer->weights) {
        errx(1, "malloc failed");
    }
    init_matrix(layer->weights);
    layer->weights->data = (vector_t**)malloc(output_size * sizeof(vector_t*));
    if (!layer->weights->data) {
        errx(1, "malloc failed");
    }
    layer->weights->size = output_size;
    layer->weights->capacity = output_size;

    for (size_t i = 0; i < output_size; i++) {
        layer->weights->data[i] = (vector_t*)malloc(sizeof(vector_t));
        if (!layer->weights->data[i]) {
            errx(1, "malloc failed");
        }
        init_vector(layer->weights->data[i]);
        layer->weights->data[i]->data = (float*)malloc(input_size * sizeof(float));
        if (!layer->weights->data[i]->data) {
            errx(1, "malloc failed");
        }
        layer->weights->data[i]->size = input_size;
        layer->weights->data[i]->capacity = input_size;

        for (size_t j = 0; j < input_size; j++) {
            float r = (float)rand() / (float)RAND_MAX;
            layer->weights->data[i]->data[j] = -limit + 2.0f * limit * r;
        }
    }

    layer->biases = (vector_t*)malloc(sizeof(vector_t));
    if (!layer->biases) {
        errx(1, "malloc failed");
    }
    init_vector(layer->biases);
    layer->biases->data = (float*)calloc(output_size, sizeof(float));
    if (!layer->biases->data) {
        errx(1, "malloc failed");
    }
    layer->biases->size = output_size;
    layer->biases->capacity = output_size;

    layer->activation = activation;

    return layer;
}

void free_layer(layer_t* layer) {
    if (!layer) {
        errx(1, "Nullptr detected");
    }

    free_matrix(layer->weights);
    free(layer->weights);
    free_vector(layer->biases);
    free(layer->biases);
    free(layer);
}

vector_t* layer_forward(const layer_t* layer, const vector_t* input) {
    if (!layer || !input || !input->data) {
        errx(1, "Nullptr detected");
    }

    vector_t* z = (vector_t*)malloc(sizeof(vector_t));
    if (!z) {
        errx(1, "malloc failed");
    }
    init_vector(z);
    z->data = (float*)malloc(layer->weights->size * sizeof(float));
    if (!z->data) {
        errx(1, "malloc failed");
    }
    z->size = layer->weights->size;
    z->capacity = layer->weights->size;

    for (size_t i = 0; i < layer->weights->size; i++) {
        z->data[i] = avx_256_vector_dot(layer->weights->data[i], input);
    }

    avx_256_vector_add(z, layer->biases, z);

    if (layer->activation) {
        vector_t* activated = layer->activation(z);
        free_vector(z);
        free(z);
        return activated;
    }

    return z;
}

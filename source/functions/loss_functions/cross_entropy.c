#include <math.h>

#include "source/functions/loss_functions/cross_entropy.h"

#define EPSILON 1e-7f

float calculate_cross_entropy(const vector_t* predictions, const vector_t* y) {
    if (!predictions || !y || !predictions->data || !y->data || predictions->size != y->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    float loss = 0.0f;
    for (size_t i = 0; i < y->size; i++) {
        float p = predictions->data[i];
        if (p < EPSILON) p = EPSILON;
        loss -= y->data[i] * logf(p);
    }

    return loss;
}

// Combined softmax + cross-entropy delta: predictions - y
// This is the gradient of cross-entropy w.r.t. the pre-softmax logits.
// Use this as the initial delta in network_backward instead of (output - y).
vector_t* cross_entropy_softmax_delta(const vector_t* predictions, const vector_t* y) {
    if (!predictions || !y || !predictions->data || !y->data || predictions->size != y->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    vector_t* delta = (vector_t*)malloc(sizeof(vector_t));
    if (!delta) {
        errx(1, "Malloc operation failed");
    }
    init_vector(delta);
    copy_vector(delta, predictions);
    avx_256_vector_sub(delta, y, delta);

    return delta;
}

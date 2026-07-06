#include "source/network/forward_cache.h"

forward_cache_t* init_forward_cache(size_t count) {
    forward_cache_t* cache = (forward_cache_t*)malloc(sizeof(forward_cache_t));
    if (!cache) {
        errx(1, "Malloc operation failed");
    }

    cache->activations = (vector_t**)malloc(count * sizeof(vector_t*));
    if (!cache->activations) {
        errx(1, "Malloc operation failed");
    }

    cache->pre_activations = (vector_t**)malloc(count * sizeof(vector_t*));
    if (!cache->pre_activations) {
        errx(1, "Malloc operation failed");
    }

    for (size_t i = 0; i < count; i++) {
        cache->activations[i] = NULL;
        cache->pre_activations[i] = NULL;
    }

    cache->count = count;
    return cache;
}

void free_forward_cache(forward_cache_t* cache) {
    if (!cache) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < cache->count; i++) {
        if (cache->activations[i]) {
            free_vector(cache->activations[i]);
            free(cache->activations[i]);
        }
        if (cache->pre_activations[i]) {
            free_vector(cache->pre_activations[i]);
            free(cache->pre_activations[i]);
        }
    }

    free(cache->activations);
    free(cache->pre_activations);
    free(cache);
}

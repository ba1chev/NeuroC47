#include "network/forward_cache.h"

forward_cache_t* init_forward_cache(size_t count) {
    forward_cache_t* cache = (forward_cache_t*)malloc(sizeof(forward_cache_t));
    if (!cache) {
        errx(1, "malloc failed");
    }

    cache->activations = (vector_t**)malloc(count * sizeof(vector_t*));
    if (!cache->activations) {
        errx(1, "malloc failed");
    }

    for (size_t i = 0; i < count; i++) {
        cache->activations[i] = NULL;
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
    }

    free(cache->activations);
    free(cache);
}

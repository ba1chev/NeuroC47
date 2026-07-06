#ifndef FORWARD_CACHE_H
#define FORWARD_CACHE_H

#include "source/structures/vector/vector.h"

typedef struct forward_cache_t {
    vector_t** activations;
    vector_t** pre_activations;
    size_t count;
} forward_cache_t;

forward_cache_t* init_forward_cache(size_t count);
void free_forward_cache(forward_cache_t* cache);

#endif

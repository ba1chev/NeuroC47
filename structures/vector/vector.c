#include <string.h>
#include <immintrin.h>

#include "structures/vector/vector.h"

void init_vector(vector_t* vector) {
    if (!vector) {
        errx(1, "Nullptr detected");
    }

    vector->data = NULL;
    vector->size = 0;
    vector->capacity = 0;
}

void free_vector(vector_t* vector) {
    if (!vector) {
        errx(1, "Nullptr detected");
    }

    free(vector->data);
    vector->data = NULL;
    vector->size = 0;
    vector->capacity = 0;
}

void push_vector(vector_t* vector, const float* new_element) {
    if (!vector || !new_element) {
        errx(1, "Nullptr detected");
    }

    if (vector->size == vector->capacity) {
        size_t new_capacity = vector->capacity == 0 ? 1 : vector->capacity * 2;
        float* new_data = (float*)malloc(sizeof(float) * new_capacity);
        if (!new_data) {
            errx(1, "malloc failed");
        }

        if (vector->size > 0) {
            memcpy(new_data, vector->data, sizeof(float) * vector->size);
        }

        free(vector->data);
        vector->data = new_data;
        vector->capacity = new_capacity;
    }

    vector->data[vector->size] = *new_element;
    vector->size += 1;
}

void pop_vector(vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        errx(1, "Nullptr detected or the vector is already empty");
    }

    vector->size -= 1;
}

void copy_vector(vector_t* __restrict left_vector, const vector_t* __restrict right_vector) {
    if (!left_vector || !right_vector || left_vector == right_vector) {
        errx(1, "Nullptr detected or aliasing");
    }

    free(left_vector->data);
    left_vector->size = right_vector->size;
    left_vector->capacity = right_vector->capacity;

    if (right_vector->capacity == 0) {
        left_vector->data = NULL;
    } else {
        left_vector->data = (float*)malloc(sizeof(float) * right_vector->capacity);
        if (!left_vector->data) {
            errx(1, "malloc failed");
        }
        if (right_vector->size > 0) {
            memcpy(left_vector->data, right_vector->data, sizeof(float) * right_vector->size);
        }
    }
}

void avx_256_vector_hadamard(
    const vector_t* __restrict left,
    const vector_t* __restrict right,
    vector_t* __restrict result
) {
    if (!left || !right || !result
        || !left->data || !right->data
        || left->size != right->size) {
        errx(1, "Nullptr detected or size is not equal");
    }

    if (result->capacity < left->size) {
        free(result->data);
        result->data = (float*)malloc(sizeof(float) * left->size);
        if (!result->data) {
            errx(1, "malloc failed");
        }
        result->capacity = left->size;
    }
    result->size = left->size;

    size_t current_index = 0;
    size_t simd_end = left->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 left_avx_vector = _mm256_loadu_ps(&left->data[current_index]);
        __m256 right_avx_vector = _mm256_loadu_ps(&right->data[current_index]);
        __m256 mult_avx_vector = _mm256_mul_ps(left_avx_vector, right_avx_vector);
        _mm256_storeu_ps(&result->data[current_index], mult_avx_vector);
    }

    for (; current_index < left->size; current_index += 1) {
        result->data[current_index] = left->data[current_index] * right->data[current_index];
    }
}

void avx_256_vector_scalar_mult(
    vector_t* __restrict vector,
    const float scalar
) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;
    __m256 scalar_avx_vector = _mm256_set1_ps(scalar);

    for (; current_index < simd_end; current_index += 8) {
        __m256 data_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        __m256 mult_avx_vector = _mm256_mul_ps(data_avx_vector, scalar_avx_vector);
        _mm256_storeu_ps(&vector->data[current_index], mult_avx_vector);
    }

    for (; current_index < vector->size; current_index += 1) {
        vector->data[current_index] *= scalar;
    }
}

void avx_256_vector_zero(vector_t* vector) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 zero_avx_vector = _mm256_setzero_ps();
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        _mm256_storeu_ps(&vector->data[current_index], zero_avx_vector);
    }

    for (; current_index < vector->size; current_index += 1) {
        vector->data[current_index] = 0.0f;
    }
}

void avx_256_vector_fill(vector_t* vector, const float value) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 value_avx_vector = _mm256_set1_ps(value);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        _mm256_storeu_ps(&vector->data[current_index], value_avx_vector);
    }

    for (; current_index < vector->size; current_index += 1) {
        vector->data[current_index] = value;
    }
}

static void ensure_result_capacity(vector_t* result, size_t required_size) {
    if (result->capacity < required_size) {
        free(result->data);
        result->data = (float*)malloc(sizeof(float) * required_size);
        if (!result->data) {
            errx(1, "Malloc failed");
        }
        result->capacity = required_size;
    }
    result->size = required_size;
}

void avx_256_vector_add(
    const vector_t* __restrict left,
    const vector_t* __restrict right,
    vector_t* __restrict result
) {
    if (!left || !right || !result
        || !left->data || !right->data
        || left->size != right->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    ensure_result_capacity(result, left->size);

    size_t current_index = 0;
    size_t simd_end = left->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 left_avx_vector = _mm256_loadu_ps(&left->data[current_index]);
        __m256 right_avx_vector = _mm256_loadu_ps(&right->data[current_index]);
        _mm256_storeu_ps(&result->data[current_index], _mm256_add_ps(left_avx_vector, right_avx_vector));
    }

    for (; current_index < left->size; current_index += 1) {
        result->data[current_index] = left->data[current_index] + right->data[current_index];
    }
}

void avx_256_vector_sub(
    const vector_t* __restrict left,
    const vector_t* __restrict right,
    vector_t* __restrict result
) {
    if (!left || !right || !result
        || !left->data || !right->data
        || left->size != right->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    ensure_result_capacity(result, left->size);

    size_t current_index = 0;
    size_t simd_end = left->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 left_avx_vector = _mm256_loadu_ps(&left->data[current_index]);
        __m256 right_avx_vector = _mm256_loadu_ps(&right->data[current_index]);
        _mm256_storeu_ps(&result->data[current_index], _mm256_sub_ps(left_avx_vector, right_avx_vector));
    }

    for (; current_index < left->size; current_index += 1) {
        result->data[current_index] = left->data[current_index] - right->data[current_index];
    }
}

void avx_256_vector_div(
    const vector_t* __restrict left,
    const vector_t* __restrict right,
    vector_t* __restrict result
) {
    if (!left || !right || !result
        || !left->data || !right->data
        || left->size != right->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    for (size_t i = 0; i < right->size; i++) {
        if (right->data[i] == 0.0f) {
            errx(1, "Division by zero at index %zu", i);
        }
    }

    ensure_result_capacity(result, left->size);

    size_t current_index = 0;
    size_t simd_end = left->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 left_avx_vector = _mm256_loadu_ps(&left->data[current_index]);
        __m256 right_avx_vector = _mm256_loadu_ps(&right->data[current_index]);
        _mm256_storeu_ps(&result->data[current_index], _mm256_div_ps(left_avx_vector, right_avx_vector));
    }

    for (; current_index < left->size; current_index += 1) {
        result->data[current_index] = left->data[current_index] / right->data[current_index];
    }
}

void avx_256_vector_fma(
    const vector_t* __restrict a,
    const vector_t* __restrict b,
    const vector_t* __restrict c,
    vector_t* __restrict result
) {
    if (!a || !b || !c || !result
        || !a->data || !b->data || !c->data
        || a->size != b->size || a->size != c->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    ensure_result_capacity(result, a->size);

    size_t current_index = 0;
    size_t simd_end = a->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 a_avx_vector = _mm256_loadu_ps(&a->data[current_index]);
        __m256 b_avx_vector = _mm256_loadu_ps(&b->data[current_index]);
        __m256 c_avx_vector = _mm256_loadu_ps(&c->data[current_index]);
        __m256 fma_avx_vector = _mm256_add_ps(_mm256_mul_ps(a_avx_vector, b_avx_vector), c_avx_vector);
        _mm256_storeu_ps(&result->data[current_index], fma_avx_vector);
    }

    for (; current_index < a->size; current_index += 1) {
        result->data[current_index] = a->data[current_index] * b->data[current_index] + c->data[current_index];
    }
}

void avx_256_vector_scalar_add(vector_t* __restrict vector, const float scalar) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;
    __m256 value_avx_vector = _mm256_set1_ps(scalar);

    for (; current_index < simd_end; current_index += 8) {
        __m256 avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_add_ps(avx_vector, value_avx_vector));
    }

    for (; current_index < vector->size; current_index += 1) {
        vector->data[current_index] += scalar;
    }
}

void avx_256_vector_axpy(
    vector_t* __restrict target,
    const vector_t* __restrict source,
    const float scalar
) {
    // todo
}

void avx_256_vector_abs(vector_t* vector) {
    // todo
}

void avx_256_vector_sqrt(vector_t* vector) {
    // todo
}

void avx_256_vector_exp(vector_t* vector) {
    // todo
}

void avx_256_vector_clip(vector_t* vector, const float min_value, const float max_value) {
    // todo
}

void avx_256_vector_relu(vector_t* vector) {
    // todo
}

void avx_256_vector_relu_derivative(vector_t* vector) {
    // todo
}

void avx_256_vector_leaky_relu(vector_t* vector, const float alpha) {
    // todo
}

void avx_256_vector_sigmoid(vector_t* vector) {
    // todo
}

void avx_256_vector_tanh(vector_t* vector) {
    // todo
}

void avx_256_vector_softmax(vector_t* vector) {
    // todo
}

float avx_256_vector_sum(const vector_t* vector) {
    // todo
}

float avx_256_vector_sum_squares(const vector_t* vector) {
    // todo
}

float avx_256_vector_l2_norm(const vector_t* vector) {
    // todo
}

float avx_256_vector_dot(
    const vector_t* __restrict left,
    const vector_t* __restrict right
) {
    // todo
}

float avx_256_vector_min(const vector_t* vector) {
    // todo
}

float avx_256_vector_max(const vector_t* vector) {
    // todo
}

size_t avx_256_vector_argmax(const vector_t* vector) {
    // todo
}

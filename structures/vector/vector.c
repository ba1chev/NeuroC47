#include <math.h>
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
        __m256 fma_avx_vector = _mm256_fmadd_ps(a_avx_vector, b_avx_vector, c_avx_vector);
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
    if (!target || !source || !target->data || !source->data
        || target->size != source->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    size_t current_index = 0;
    size_t simd_end = target->size & ~(size_t)7;
    __m256 scalar_avx_vector = _mm256_set1_ps(scalar);

    for (; current_index < simd_end; current_index += 8) {
        __m256 source_avx_vector = _mm256_loadu_ps(&source->data[current_index]);
        __m256 target_avx_vector = _mm256_loadu_ps(&target->data[current_index]);
        _mm256_storeu_ps(&target->data[current_index], _mm256_fmadd_ps(scalar_avx_vector, source_avx_vector, target_avx_vector));
    }

    for (; current_index < target->size; current_index++) {
        target->data[current_index] += scalar * source->data[current_index];
    }
}

void avx_256_vector_abs(vector_t* vector) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 v = _mm256_loadu_ps(&vector->data[current_index]);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_andnot_ps(sign_mask, v));
    }

    for (; current_index < vector->size; current_index++) {
        vector->data[current_index] = fabsf(vector->data[current_index]);
    }
}

void avx_256_vector_sqrt(vector_t* vector) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 v = _mm256_loadu_ps(&vector->data[current_index]);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_sqrt_ps(v));
    }

    for (; current_index < vector->size; current_index++) {
        vector->data[current_index] = sqrtf(vector->data[current_index]);
    }
}

void avx_256_vector_exp(vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        float* data = &vector->data[current_index];
        data[0] = expf(data[0]);
        data[1] = expf(data[1]);
        data[2] = expf(data[2]);
        data[3] = expf(data[3]);
        data[4] = expf(data[4]);
        data[5] = expf(data[5]);
        data[6] = expf(data[6]);
        data[7] = expf(data[7]);
    }

    for (; current_index < vector->size; current_index++) {
        vector->data[current_index] = expf(vector->data[current_index]);
    }
}

void avx_256_vector_clip(vector_t* vector, const float min_value, const float max_value) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 min_avx = _mm256_set1_ps(min_value);
    __m256 max_avx = _mm256_set1_ps(max_value);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_min_ps(_mm256_max_ps(current_avx_vector, min_avx), max_avx));
    }

    for (; current_index < vector->size; current_index++) {
        vector->data[current_index] = fminf(fmaxf(vector->data[current_index], min_value), max_value);
    }
}

void avx_256_vector_relu(vector_t* vector) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 zero = _mm256_setzero_ps();
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 v = _mm256_loadu_ps(&vector->data[current_index]);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_max_ps(v, zero));
    }

    for (; current_index < vector->size; current_index++) {
        if (vector->data[current_index] < 0.0f) vector->data[current_index] = 0.0f;
    }
}

void avx_256_vector_relu_derivative(vector_t* vector) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        __m256 mask = _mm256_cmp_ps(current_avx_vector, zero, _CMP_GT_OQ);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_and_ps(mask, one));
    }

    for (; current_index < vector->size; current_index++) {
        vector->data[current_index] = vector->data[current_index] > 0.0f ? 1.0f : 0.0f;
    }
}

void avx_256_vector_leaky_relu(vector_t* vector, const float alpha) {
    if (!vector || (vector->size > 0 && !vector->data)) {
        errx(1, "Nullptr detected");
    }

    __m256 zero = _mm256_setzero_ps();
    __m256 alpha_avx = _mm256_set1_ps(alpha);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        __m256 mask = _mm256_cmp_ps(current_avx_vector, zero, _CMP_GT_OQ);
        __m256 neg_avx_vector = _mm256_mul_ps(current_avx_vector, alpha_avx);
        _mm256_storeu_ps(&vector->data[current_index], _mm256_blendv_ps(neg_avx_vector, current_avx_vector, mask));
    }

    for (; current_index < vector->size; current_index++) {
        float x = vector->data[current_index];
        vector->data[current_index] = x > 0.0f ? x : alpha * x;
    }
}

void avx_256_vector_sigmoid(vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < vector->size; i++) {
        vector->data[i] = 1.0f / (1.0f + expf(-vector->data[i]));
    }
}

void avx_256_vector_tanh(vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < vector->size; i++) {
        vector->data[i] = tanhf(vector->data[i]);
    }
}

void avx_256_vector_softmax(vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        errx(1, "Nullptr detected or empty vector");
    }

    float max_val = avx_256_vector_max(vector);
    avx_256_vector_scalar_add(vector, -max_val);
    avx_256_vector_exp(vector);
    float sum = avx_256_vector_sum(vector);
    avx_256_vector_scalar_mult(vector, 1.0f / sum);
}

float avx_256_vector_sum(const vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;
    __m256 scaler_avx_vector = _mm256_setzero_ps();

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        scaler_avx_vector = _mm256_add_ps(scaler_avx_vector, current_avx_vector);
    }
    
    float result = 0.0f;
    float temp_buffer[8];
    _mm256_storeu_ps(temp_buffer, scaler_avx_vector);

    for (size_t i = 0; i < 8; i++) {
        result += temp_buffer[i];
    }
    
    for (; current_index < vector->size; current_index++) {
        result += vector->data[current_index];    
    }
    
    return result;
}

float avx_256_vector_sum_squares(const vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }

    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;
    __m256 scaler_avx_vector = _mm256_setzero_ps();

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        __m256 squred_avx_vector = _mm256_mul_ps(current_avx_vector, current_avx_vector);
        scaler_avx_vector = _mm256_add_ps(scaler_avx_vector, squred_avx_vector);
    }
    
    float result = 0.0f;
    float temp_buffer[8];
    _mm256_storeu_ps(temp_buffer, scaler_avx_vector);

    for (size_t i = 0; i < 8; i++) {
        result += temp_buffer[i];
    }
    
    for (; current_index < vector->size; current_index++) {
        result += vector->data[current_index] * vector->data[current_index];
    }
    
    return result;
}

float avx_256_vector_l2_norm(const vector_t* vector) {
    return sqrtf(avx_256_vector_sum_squares(vector));
}

float avx_256_vector_dot(
    const vector_t* __restrict left,
    const vector_t* __restrict right
) {
    if (!left || !right || !left->data || !right->data) {
        errx(1, "Null pointer detected");
    }

    if (left->size != right->size) {
        errx(1, "Vectors must have the same size");
    }

    __m256 result_vector = _mm256_setzero_ps();
    size_t simd_end = left->size & ~(size_t)7;
    size_t current_index = 0;

    for (; current_index < simd_end; current_index += 8) {
        __m256 left_vector  = _mm256_loadu_ps(&left->data[current_index]);
        __m256 right_vector = _mm256_loadu_ps(&right->data[current_index]);
        __m256 product = _mm256_mul_ps(left_vector, right_vector);
        result_vector = _mm256_add_ps(result_vector, product);
    }

    float temp_buffer[8];
    float result = 0.0f;
    _mm256_storeu_ps(temp_buffer, result_vector);

    for (int j = 0; j < 8; j++) {
        result += temp_buffer[j];
    }

    for (; current_index < left->size; current_index++) {
        result += left->data[current_index] * right->data[current_index];
    }

    return result;
}

float avx_256_vector_min(const vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }
    if (vector->size == 0) {
        errx(1, "Vector is empty");
    }

    __m256 min_avx_vector = _mm256_set1_ps(INFINITY);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        min_avx_vector = _mm256_min_ps(min_avx_vector, current_avx_vector);
    }

    float result = INFINITY;
    float temp_buffer[8];
    _mm256_storeu_ps(temp_buffer, min_avx_vector);

    for (size_t i = 0; i < 8; i++) {
        result = fminf(result, temp_buffer[i]);
    }
    
    for (; current_index < vector->size; current_index++) {
        result = fminf(result, vector->data[current_index]);
    }
    
    return result; 
}

float avx_256_vector_max(const vector_t* vector) {
    if (!vector || !vector->data) {
        errx(1, "Nullptr detected");
    }
    if (vector->size == 0) {
        errx(1, "Vector is empty");
    }

    __m256 max_avx_vector = _mm256_set1_ps(-INFINITY);
    size_t current_index = 0;
    size_t simd_end = vector->size & ~(size_t)7;

    for (; current_index < simd_end; current_index += 8) {
        __m256 current_avx_vector = _mm256_loadu_ps(&vector->data[current_index]);
        max_avx_vector = _mm256_max_ps(max_avx_vector, current_avx_vector);
    }

    float result = -INFINITY;
    float temp_buffer[8];
    _mm256_storeu_ps(temp_buffer, max_avx_vector);

    for (size_t i = 0; i < 8; i++) {
        result = fmaxf(result, temp_buffer[i]);
    }
    
    for (; current_index < vector->size; current_index++) {
        result = fmaxf(result, vector->data[current_index]);
    }
    
    return result;
}

size_t avx_256_vector_argmax(const vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        errx(1, "Nullptr detected or empty vector");
    }

    size_t best_index = 0;
    float best_value = vector->data[0];

    for (size_t i = 1; i < vector->size; i++) {
        if (vector->data[i] > best_value) {
            best_value = vector->data[i];
            best_index = i;
        }
    }

    return best_index;
}

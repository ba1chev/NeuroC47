#ifndef VECTOR_H
#define VECTOR_H

#include <err.h>
#include <stdlib.h>
#include <sys/types.h>

typedef struct vector_t {
    float* data;
    size_t size;
    size_t capacity;
} vector_t;

// Base lifecycle
void init_vector(vector_t* vector);
void free_vector(vector_t* vector);
void copy_vector(vector_t* __restrict left_vector, const vector_t* __restrict right_vector);
void push_vector(vector_t* vector, const float* new_element);
void pop_vector(vector_t* vector);

// AVX-256 SIMD operations
// --- Buffer initialization (in-place) ---
void avx_256_vector_zero(vector_t* vector);
void avx_256_vector_fill(vector_t* vector, const float value);

// --- Elementwise arithmetic, vector op vector (out-of-place, writes "result") ---
void avx_256_vector_add(const vector_t* __restrict left, const vector_t* __restrict right, vector_t* __restrict result);
void avx_256_vector_sub(const vector_t* __restrict left, const vector_t* __restrict right, vector_t* __restrict result);
void avx_256_vector_hadamard(const vector_t* __restrict left, const vector_t* __restrict right, vector_t* __restrict result);
void avx_256_vector_div(const vector_t* __restrict left, const vector_t* __restrict right, vector_t* __restrict result);
void avx_256_vector_fma(const vector_t* __restrict a, const vector_t* __restrict b, const vector_t* __restrict c, vector_t* __restrict result);

// --- Elementwise arithmetic, vector op scalar (in-place) ---
void avx_256_vector_scalar_add(vector_t* __restrict vector, const float scalar);
void avx_256_vector_scalar_mult(vector_t* __restrict vector, const float scalar);

// --- Fused ops for training loops (in-place on "target") ---
void avx_256_vector_axpy(vector_t* __restrict target, const vector_t* __restrict source, const float scalar);

// --- Elementwise unary math (in-place) ---
void avx_256_vector_abs(vector_t* vector);
void avx_256_vector_sqrt(vector_t* vector);
void avx_256_vector_exp(vector_t* vector);
void avx_256_vector_clip(vector_t* vector, const float min_value, const float max_value);

// --- Activation functions (in-place) ---
void avx_256_vector_relu(vector_t* vector);
void avx_256_vector_relu_derivative(vector_t* vector);
void avx_256_vector_leaky_relu(vector_t* vector, const float alpha);
void avx_256_vector_sigmoid(vector_t* vector);
void avx_256_vector_tanh(vector_t* vector);
void avx_256_vector_softmax(vector_t* vector);

// --- Reductions (return scalar) ---
float avx_256_vector_sum(const vector_t* vector);
float avx_256_vector_sum_squares(const vector_t* vector);
float avx_256_vector_l2_norm(const vector_t* vector);
float avx_256_vector_dot(const vector_t* __restrict left, const vector_t* __restrict right);
float avx_256_vector_min(const vector_t* vector);
float avx_256_vector_max(const vector_t* vector);
size_t avx_256_vector_argmax(const vector_t* vector);

#endif

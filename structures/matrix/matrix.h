#ifndef MATRIX_H
#define MATRIX_H

#include "structures/vector/vector.h"

typedef struct matrix_t {
    vector_t** data;
    size_t size;
    size_t capacity;
} matrix_t;

// Base lifecycle
void init_matrix(matrix_t* matrix);
void free_matrix(matrix_t* matrix);
void copy_matrix(matrix_t* __restrict left_matrix, const matrix_t* __restrict right_matrix);
void push_matrix(matrix_t* matrix, const vector_t* new_row);
void pop_matrix(matrix_t* matrix);

// Matrix operations
matrix_t* matrix_multiplication(const matrix_t* __restrict left, const matrix_t* __restrict right);
matrix_t* matrix_addition(const matrix_t* __restrict left, const matrix_t* __restrict right);
matrix_t* matrix_devision(const matrix_t* __restrict left, const matrix_t* __restrict right);
matrix_t* matrix_subtracting(const matrix_t* __restrict left, const matrix_t* __restrict right);

#endif
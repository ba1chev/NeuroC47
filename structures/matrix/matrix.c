#include "structures/matrix/matrix.h"
#include "structures/vector/vector.h"

void init_matrix(matrix_t* matrix) {
    if (!matrix) {
        errx(1, "Nullptr detected");
    }

    matrix->data = NULL;
    matrix->size = 0;
    matrix->capacity = 0;
}

void free_matrix(matrix_t* matrix) {
    if (!matrix || !matrix->data) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < matrix->size; i++) {
        free_vector(matrix->data[i]);
        free(matrix->data[i]);
    }

    free(matrix->data);
    matrix->data = NULL;
    matrix->size = 0;
    matrix->capacity = 0;
}

void copy_matrix(matrix_t* __restrict left_matrix, const matrix_t* __restrict right_matrix) {
    if (!left_matrix || !right_matrix || !left_matrix->data || !right_matrix->data) {
        errx(1, "Nullptr detected");
    }

    if (left_matrix->size != right_matrix->size
        || left_matrix->data[0]->size != right_matrix->data[0]->size) {
        errx(1, "Not equal count of rows or not equal count of cols");
    }

    for (size_t i = 0; i < left_matrix->size; i++) {
        copy_vector(left_matrix->data[i], right_matrix->data[i]);
    }
}

void push_matrix(matrix_t* matrix, const vector_t* new_row) {
    if (!matrix || !new_row || !new_row->data) {
        errx(1, "Nullptr detected");
    }

    if (matrix->size == matrix->capacity) {
        size_t new_capacity = matrix->capacity == 0 ? 1 : matrix->capacity * 2;
        vector_t** new_data = (vector_t**)malloc(new_capacity * sizeof(vector_t*));
        
        if (!new_data) {
            errx(1, "Malloc operation failed");
        }

        for (size_t i = 0; i < matrix->size; i++) {
            new_data[i] = matrix->data[i];
        }

        free(matrix->data);
        matrix->data = new_data;
        matrix->capacity = new_capacity;
    }

    matrix->data[matrix->size] = (vector_t*)malloc(sizeof(vector_t));
    if (!matrix->data[matrix->size]) {
        errx(1, "malloc failed");
    }
    init_vector(matrix->data[matrix->size]);
    copy_vector(matrix->data[matrix->size], new_row);
    matrix->size += 1;
}

void pop_matrix(matrix_t* matrix) {
    if (!matrix || !matrix->data || !matrix->size) {
        errx(1, "Nullptr detected or empty matrix");
    }

    free_vector(matrix->data[matrix->size - 1]);
    free(matrix->data[matrix->size - 1]);
    matrix->data[matrix->size - 1] = NULL;
    matrix->size -= 1;
}

matrix_t* matrix_multiplication(const matrix_t* __restrict left, const matrix_t* __restrict right) {
    // todo
}

matrix_t* matrix_addition(const matrix_t* __restrict left, const matrix_t* __restrict right) {
    if (!left || !right || !left->data || !right->data || left->size != right->size 
        || left->data[0]->size != right->data[0]->size) {
        errx(1, "Nullptr detected or not equal count of rows/cols");
    }

    size_t count_of_rows = left->size;
    matrix_t* result = (matrix_t*)malloc(sizeof(matrix_t));
    result->data = (vector_t**)malloc(count_of_rows * sizeof(vector_t*));
    result->size = left->size;
    result->capacity = left->capacity;

    for (size_t i = 0; i < count_of_rows; i++) {
        result->data[i] = (vector_t*)malloc(sizeof(vector_t));
        init_vector(result->data[i]);
        avx_256_vector_add(left->data[i], right->data[i], result->data[i]);
    }
    
    return result;
}

matrix_t* matrix_devision(const matrix_t* __restrict left, const matrix_t* __restrict right) {
    // todo
}

matrix_t* matrix_subtracting(const matrix_t* __restrict left, const matrix_t* __restrict right) {
    // todo
}
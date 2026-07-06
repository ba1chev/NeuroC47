#include "source/structures/matrix/matrix.h"
#include "source/structures/vector/vector.h"

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
    if (!left_matrix || !right_matrix || !right_matrix->data) {
        errx(1, "Nullptr detected");
    }

    for (size_t i = 0; i < left_matrix->size; i++) {
        free_vector(left_matrix->data[i]);
        free(left_matrix->data[i]);
    }
    free(left_matrix->data);

    left_matrix->data = (vector_t**)malloc(right_matrix->size * sizeof(vector_t*));
    if (!left_matrix->data) {
        errx(1, "malloc failed");
    }
    left_matrix->size = right_matrix->size;
    left_matrix->capacity = right_matrix->size;

    for (size_t i = 0; i < right_matrix->size; i++) {
        left_matrix->data[i] = (vector_t*)malloc(sizeof(vector_t));
        if (!left_matrix->data[i]) {
            errx(1, "malloc failed");
        }
        init_vector(left_matrix->data[i]);
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
    if (!left || !right || !left->data || !right->data
        || left->data[0]->size != right->size) {
        errx(1, "Nullptr detected or invalid dims");
    }

    size_t count_of_rows = left->size;
    size_t count_of_cols = right->data[0]->size;

    matrix_t* result = (matrix_t*)malloc(sizeof(matrix_t));
    if (!result) {
        errx(1, "Malloc operation failed");
    }

    result->data = (vector_t**)malloc(count_of_rows * sizeof(vector_t*));
    if (!result->data) {
        errx(1, "Malloc operation failed");
    }

    result->size = count_of_rows;
    result->capacity = count_of_rows;

    for (size_t i = 0; i < count_of_rows; i++) {
        result->data[i] = (vector_t*)malloc(sizeof(vector_t));
        if (!result->data[i]) {
            errx(1, "Malloc operation failed");
        }

        init_vector(result->data[i]);
        result->data[i]->data = (float*)malloc(count_of_cols * sizeof(float));

        if (!result->data[i]->data) {
            errx(1, "Malloc operation failed");
        }
        result->data[i]->size = count_of_cols;
        result->data[i]->capacity = count_of_cols;
    }

    matrix_t* right_transposed = matrix_transpose(right);

    for (size_t i = 0; i < count_of_rows; i++) {
        for (size_t j = 0; j < count_of_cols; j++) {
            result->data[i]->data[j] = avx_256_vector_dot(left->data[i], right_transposed->data[j]);
        }
    }

    free_matrix(right_transposed);
    free(right_transposed);

    return result;
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

matrix_t* matrix_transpose(const matrix_t* matrix) {
    if (!matrix || !matrix->data) {
        errx(1, "Nullptr detected");
    }

    size_t count_of_rows = matrix->size;
    size_t count_of_cols = matrix->data[0]->size;

    matrix_t* result = (matrix_t*)malloc(sizeof(matrix_t));
    if (!result) {
        errx(1, "Malloc operation failed");
    }

    result->data = (vector_t**)malloc(count_of_cols * sizeof(vector_t*));
    if (!result->data) {
        errx(1, "Malloc operation failed");
    }

    result->size = count_of_cols;
    result->capacity = count_of_cols;

    for (size_t j = 0; j < count_of_cols; j++) {
        result->data[j] = (vector_t*)malloc(sizeof(vector_t));
        if (!result->data[j]) {
            errx(1, "Malloc operation failed");
        }

        init_vector(result->data[j]);
        result->data[j]->data = (float*)malloc(count_of_rows * sizeof(float));
        if (!result->data[j]->data) {
            errx(1, "Malloc operation failed");
        }
        
        result->data[j]->size = count_of_rows;
        result->data[j]->capacity = count_of_rows;

        for (size_t i = 0; i < count_of_rows; i++) {
            result->data[j]->data[i] = matrix->data[i]->data[j];
        }
    }

    return result;
}

matrix_t* matrix_subtracting(const matrix_t* __restrict left, const matrix_t* __restrict right) {
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
        avx_256_vector_sub(left->data[i], right->data[i], result->data[i]);
    }
    
    return result;
}
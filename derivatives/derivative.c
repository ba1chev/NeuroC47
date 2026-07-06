#include "derivatives/derivative.h"

vector_t* calculate_derivative_at(const derivative_t* derivative_obj, const vector_t* X) {
    if (!derivative_obj || !derivative_obj->function) {
        errx(1, "Nullptr detected");
    }

    vector_t* result = (vector_t*)malloc(sizeof(vector_t));
    if (!result) {
        errx(1, "malloc failed");
    }
    init_vector(result);
    result->data = (float*)malloc(X->size * sizeof(float));
    if (!result->data) {
        errx(1, "malloc failed");
    }
    result->size = X->size;
    result->capacity = X->size;

    for (size_t i = 0; i < X->size; i++) {
        vector_t* left_part = (vector_t*)malloc(sizeof(vector_t));
        vector_t* right_part = (vector_t*)malloc(sizeof(vector_t));
        if (!left_part || !right_part) {
            errx(1, "malloc failed");
        }
        init_vector(left_part);
        init_vector(right_part);
        copy_vector(left_part, X);
        copy_vector(right_part, X);

        left_part->data[i]  += derivative_obj->precision;
        right_part->data[i] -= derivative_obj->precision;

        vector_t* left_val  = derivative_obj->function(left_part);
        vector_t* right_val = derivative_obj->function(right_part);

        result->data[i] = (left_val->data[i] - right_val->data[i]) / (2.0f * derivative_obj->precision);

        free_vector(left_val);   free(left_val);
        free_vector(right_val);  free(right_val);
        free_vector(left_part);  free(left_part);
        free_vector(right_part); free(right_part);
    }

    return result;
}
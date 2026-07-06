#include <math.h>

#include "source/functions/loss_functions/mean_squared_error.h"

float calculate_mse(function_t fn_predicate, const vector_t* X, const vector_t* y) {
    if (!fn_predicate || !X || !y || !X->data || !y->data || X->size != y->size) {
        errx(1, "Nullptr detected or size mismatch");
    }

    vector_t* predictions = fn_predicate(X);

    float result = 0.0f;
    for (size_t i = 0; i < y->size; i++) {
        float diff = predictions->data[i] - y->data[i];
        result += diff * diff;
    }

    free_vector(predictions);
    free(predictions);

    return result / y->size;
}
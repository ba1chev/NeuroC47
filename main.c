#include <err.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

typedef struct matrix_t {
    float number;
} matrix_t;


int main(void) {
    matrix_t matrix;
    matrix.number = 3;

    if (printf("%.3f/n", matrix.number) < 0) {
        err(1, "Printf error");
    } 

    return 0;
}
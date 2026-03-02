#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu_stats.h"

/* default size */
#define DEFAULT_SIZE 1024

/* Processor frequency in GHZ */
#define PROC_FREQ 2.4

#define NB_EXPERIMENTS 102

#define CHUNK 4

typedef double *vector;
typedef double *matrix;

static struct cpu_stats_report experiments[NB_EXPERIMENTS];

void init_vector(vector *X, const size_t size, const double val) {
    unsigned int i = 0;

    *X = malloc(sizeof(double) * size);

    if (*X == NULL) {
        perror("vector allocation");
        exit(-1);
    }

    for (i = 0; i < size; i++)
        (*X)[i] = val;

    return;
}

void free_vector(vector X) {
    free(X);

    return;
}

void init_matrix_inf(matrix *X, const size_t size, double val) {
    unsigned int i = 0, j = 0;
    *X = malloc(sizeof(double) * size * size);

    if (*X == NULL) {
        perror("matrix allocation");
        exit(-1);
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < i; j++) {
            (*X)[i * size + j] = val;
            (*X)[j * size + i] = 0.0;
        }
        (*X)[i * size + i] = val;
    }
}

void free_matrix(matrix X) {
    free(X);

    return;
}

void print_vector(vector X, const size_t size) {
    unsigned int i;

    for (i = 0; i < size; i++)
        printf(" %3.2f", X[i]);

    printf("\n\n");
    fflush(stdout);

    return;
}

void print_matrix(matrix M, const size_t size) {
    unsigned int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf(" %3.2f ", M[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

void mult_mat_vector(matrix M, vector b, vector c, const size_t size) {
    register unsigned int i;
    register unsigned int j;
    register double r;

    for (i = 0; i < size; i = i + 1) {
        r = 0.0;
        for (j = 0; j < size; j = j + 1) {
            r += M[i * size + j] * b[j];
        }
        c[i] = r;
    }
    return;
}

void mult_mat_vector_tri_inf(matrix M, vector b, vector c, const size_t size) {
    /*
      this function is sequential (no OpenMP directive)
      Computes the Multiplication between the vector b and the Triangular Lower Matrix
    */
    return;
}

void mult_mat_vector_tri_inf1(matrix M, vector b, vector c, const size_t size) {
    /*
      this function is parallel (with OpenMP directive, static scheduling)
      Computes the Multiplication between the vector b and the Triangular Lower Matrix
    */

    return;
}

void mult_mat_vector_tri_inf2(matrix M, vector b, vector c, const size_t size) {
    /*
      this function is parallel (with OpenMP directive, dynamic scheduling)
      Computes the Multiplication between the vector b and the Triangular Lower Matrix
    */

    return;
}

void mult_mat_vector_tri_inf3(matrix M, vector b, vector c, const size_t size) {
    /*
      this function is parallel (with OpenMP directive, guided scheduling)
      Computes the Multiplication between the vector b and the Triangular Lower Matrix
    */

    return;
}

void mult_mat_vector_tri_inf4(matrix M, vector b, vector c, const size_t size) {
    /*
      this function is parallel (with OpenMP directive, runtime scheduling)
      Computes the Multiplication between the vector b and the Triangular Lower Matrix
    */

    return;
}

int main(int argc, char *argv[]) {
    matrix M;
    vector v1;
    vector v2;

    size_t datasize = DEFAULT_SIZE;

    unsigned int exp;

    if (argc > 2) {
        printf("usage: %s [data_size]\n", argv[0]);
        exit(-1);
    } else {
        if (argc == 2) {
            datasize = atoi(argv[1]);
        }
    }

    printf("Testing with Matrices of size %zu X %zu\n\n", datasize, datasize);

    printf("number of threads %d\n", omp_get_max_threads());

    // Initialize CPU counters and clocks
    struct cpu_stats *stats = cpu_stats_init();

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    /* print_vector (v1, datasize) ; */
    /* print_matrix (M, datasize) ; */

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }

    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("Full Matrix * Vector multiplication", avg_stats);
    }

    free_vector(v1);
    free_vector(v2);
    free_matrix(M);

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector_tri_inf(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("Triangular Matrix * Vector multiplication", avg_stats);
    }

    free_vector(v1);
    free_vector(v2);
    free_matrix(M);

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector_tri_inf1(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }

    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("OpenMP Parallel Loop Static Scheduling", avg_stats);
    }

    free_vector(v1);
    free_vector(v2);
    free_matrix(M);

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector_tri_inf2(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }

    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("OpenMP Parallel Loop Dynamic Scheduling", avg_stats);
    }

    free_vector(v1);
    free_vector(v2);
    free_matrix(M);

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector_tri_inf3(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("Parallel Loop Guided Scheduling", avg_stats);
    }

    free_vector(v1);
    free_vector(v2);
    free_matrix(M);

    init_vector(&v1, datasize, 1.0);
    init_matrix_inf(&M, datasize, 2.0);
    init_vector(&v2, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vector_tri_inf4(M, v1, v2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("Parallel Loop Runtime Scheduling", avg_stats);
    }

    return 0;
}

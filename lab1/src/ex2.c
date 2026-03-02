#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu_stats.h"

/* default size */
#define DEFAULT_SIZE 100

/* Processor frequency in GHZ */
#define PROC_FREQ 2.4

#define NB_EXPERIMENTS 22
static struct cpu_stats_report experiments[NB_EXPERIMENTS];

typedef double *vector;
typedef double *matrix;

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

void init_matrix(matrix *X, const size_t size, const double val) {
    unsigned int i = 0, j = 0;
    *X = malloc(sizeof(double) * size * size);

    if (*X == NULL) {
        perror("matrix allocation");
        exit(-1);
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            (*X)[i * size + j] = val;
        }
    }
}

void free_matrix(matrix X) {
    free(X);

    return;
}

void print_vectors(vector X, vector Y, const size_t size) {
    unsigned int i;

    for (i = 0; i < size; i++)
        printf(" X [%d] = %le Y [%d] = %le\n", i, X[i], i, Y[i]);

    return;
}

void add_vectors1(vector X, vector Y, vector Z, const size_t size) {
    register unsigned int i;

#pragma omp parallel for schedule(static)
    for (i = 0; i < size; i++)
        X[i] = Y[i] + Z[i];

    return;
}

void add_vectors2(vector X, vector Y, vector Z, const size_t size) {
    register unsigned int i;

#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < size; i++)
        X[i] = Y[i] + Z[i];

    return;
}

double dot1(vector X, vector Y, const size_t size) {
    register unsigned int i;
    register double dot;

    dot = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot)
    for (i = 0; i < size; i++)
        dot += X[i] * Y[i];

    return dot;
}

double dot2(vector X, vector Y, const size_t size) {
    register unsigned int i;
    register double dot;

    dot = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : dot)
    for (i = 0; i < size; i++)
        dot += X[i] * Y[i];

    return dot;
}

double dot3(vector X, vector Y, const size_t size) {
    register unsigned int i;
    register double dot;

    dot = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot)
    for (i = 0; i < size; i = i + 8) {
        dot += X[i] * Y[i];
        dot += X[i + 1] * Y[i + 1];
        dot += X[i + 2] * Y[i + 2];
        dot += X[i + 3] * Y[i + 3];

        dot += X[i + 4] * Y[i + 4];
        dot += X[i + 5] * Y[i + 5];
        dot += X[i + 6] * Y[i + 6];
        dot += X[i + 7] * Y[i + 7];
    }

    return dot;
}

void mult_mat_vect0(matrix M, vector b, double *c, size_t datasize) {
    /*
    matrix vector multiplication
    Sequential function
    */

    return;
}

void mult_mat_vect1(matrix M, vector b, vector c, size_t datasize) {
    /*
    matrix vector multiplication
    Parallel function with static loop scheduling
    */

    return;
}

void mult_mat_vect2(matrix M, vector b, vector c, size_t datasize) {
    /*
    matrix vector multiplication
    Parallel function with static loop scheduling
    unrolling internal loop
    */

    return;
}

void mult_mat_mat0(matrix A, matrix B, matrix C, size_t datasize) {
    /*
    Matrix Matrix Multiplication
    Sequential function
    */

    return;
}

void mult_mat_mat1(matrix A, matrix B, matrix C, size_t datasize) {
    /*
    Matrix Matrix Multiplication
    Parallel function with OpenMP and static scheduling
    */

    return;
}

void mult_mat_mat2(matrix A, matrix B, matrix C, size_t datasize) {
    /*
    Matrix Matrix Multiplication
    Parallel function with OpenMP and static scheduling
    Unrolling the inner loop
    */

    return;
}

int main(int argc, char *argv[]) {
    int maxnthreads;

    vector a, b, c;
    matrix M1, M2;

    size_t datasize = DEFAULT_SIZE;

    int exp;

    if (argc > 2) {
        printf("usage: %s [data_size]\n", argv[0]);
        exit(-1);
    } else {
        if (argc == 2) {
            datasize = atoi(argv[1]);
        }
    }

    printf("Testing with Vectors of size %zu -- Matrices of size %zu X %zu\n\n", datasize, datasize, datasize);

    maxnthreads = omp_get_max_threads();
    printf("Max number of threads: %d \n", maxnthreads);

    // CPU counters and clocks initialization
    struct cpu_stats *stats = cpu_stats_init();

    // Vector initialization

    init_vector(&a, datasize, 1.0);
    init_vector(&b, datasize, 2.0);
    init_vector(&c, datasize, 0.0);

    /*
    print_vectors (a, b, datasize) ;
    */

    printf("=============== ADD ==========================================\n");

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        add_vectors1(c, a, b, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("add OpenMP static loop", avg_stats);

        double flops = ( double ) datasize ; // 1 floating point operation per element (1 addition)
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ; // Converting flops / s to Gflops
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }

    /* code to print the GLOP per seconds */
    /* printf ("%3.3f GFLOP per second\n", [TODO: ADD YOUR COMPUTATION]) ; */

    free_vector(a);
    free_vector(b);
    free_vector(c);

    init_vector(&a, datasize, 1.0);
    init_vector(&b, datasize, 2.0);
    init_vector(&c, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        add_vectors2(c, a, b, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("add OpenMP dynamic loop", avg_stats);
        double flops = ( double ) datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_vector(a);
    free_vector(b);
    free_vector(c);

    printf("==============================================================\n\n");

    printf("====================DOT =====================================\n");

    init_vector(&a, datasize, 1.0);
    init_vector(&b, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        dot1(a, b, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("dot OpenMP static loop", avg_stats);
        double flops = 2.0 * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_vector(a);
    free_vector(b);

    init_vector(&a, datasize, 1.0);
    init_vector(&b, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        dot2(a, b, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("dot OpenMP dynamic loop", avg_stats);
        double flops = 2.0 * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_vector(a);
    free_vector(b);

    init_vector(&a, datasize, 1.0);
    init_vector(&b, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        dot3(a, b, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("dot OpenMP static unrolled loop", avg_stats);
        double flops = 2.0 * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_vector(a);
    free_vector(b);

    printf("=============================================================\n\n");

    printf("======================== Mult Mat Vector =====================================\n");

    init_matrix(&M1, datasize, 1.0);
    init_vector(&b, datasize, 2.0);
    init_vector(&a, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vect0(M1, b, a, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_vect Sequential", avg_stats);
        double flops = 2.0 * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_vector(a);
    free_vector(b);

    init_matrix(&M1, datasize, 1.0);
    init_vector(&b, datasize, 2.0);
    init_vector(&a, datasize, 0.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vect1(M1, b, a, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_vect OpenMP Static loop", avg_stats);
        double flops = 2.0 * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_vector(a);
    free_vector(b);

    init_matrix(&M1, datasize, 1.0);
    init_vector(&b, datasize, 2.0);
    init_vector(&a, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_vect2(M1, b, a, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_vect OpenMP Static unrolled loop", avg_stats);
        double flops = 2.0 * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_vector(a);
    free_vector(b);

    printf("===================================================================\n\n");

    printf("======================== Mult Mat Mat =====================================\n");

    init_matrix(&M1, datasize, 1.0);
    init_matrix(&M2, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_mat0(M1, M2, M2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_mat Sequential loop", avg_stats);
        double flops = 2.0 * datasize * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_matrix(M2);

    init_matrix(&M1, datasize, 1.0);
    init_matrix(&M2, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_mat1(M1, M2, M2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_mat OpenMP Static loop", avg_stats);
        double flops = 2.0 * datasize * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_matrix(M2);

    init_matrix(&M1, datasize, 1.0);
    init_matrix(&M2, datasize, 2.0);

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
        cpu_stats_begin(stats);

        mult_mat_mat2(M1, M2, M2, datasize);

        experiments[exp] = cpu_stats_end(stats);
    }
    {
        struct cpu_stats_report avg_stats = average_report(experiments, NB_EXPERIMENTS);
        println_cpu_stats_report("mult_mat_mat OpenMP Static unrolled loop", avg_stats);
        double flops = 2.0 * datasize * datasize * datasize ;
        double gflops = flops / avg_stats.elapsed_real_time_secs / 1e9 ;
        printf( "%3.3f GFLOP per second\n" , gflops ) ;
    }
    free_matrix(M1);
    free_matrix(M2);

    printf("===================================================================\n\n");

    return 0;
}

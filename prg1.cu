/*
PRG1 - Програма 1
Задача:
MA = MB * MC + d * ME

Виконав:
Скоробагатько Іван ІО-13

PRG1 - Program 1
Task:
MA = MB * MC + d * ME

Programmed by:
Skorobagatko Ivan ІО-13
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpu_kernel.cuh"
#include "cpu_math.h"
#include "file_handler.h"
#include "config.h"


void fill_matrix(int* matrix, int N);
void print_matrix_result(int* matrix, int N);
void cpu_mode();
void gpu_mode();

int main(int argc, char* argv[]) {
    // Check if the program is run with the correct number of arguments
    if (argc != 2) {
        printf("Usage: %s <GPU_ENABLE>\n", argv[0]);
        return -1;
    }

    // Parse GPU_ENABLE from command line argument
    int GPU_ENABLE = atoi(argv[1]);

    // Check if GPU_ENABLE is valid
    if (GPU_ENABLE != 0 && GPU_ENABLE != 1) {
        printf("Error: GPU_ENABLE must be 0 or 1\n");
        return -1;
    }

    // Variables


    // CPU mode memory allocation
    if (GPU_ENABLE == 0)
        cpu_mode();

    // GPU mode memory allocation
    if (GPU_ENABLE == 1)
        gpu_mode();

    printf("Done\n");
    return 0;
}

void fill_matrix(int* matrix, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i * N + j] = 0;
}

void print_matrix_result(int* matrix, int N) {
    if (N > 32) {
        printf("Result is too large to print.\n");
        return;
    }

    printf("Result: ");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", matrix[i * N + j]);
        printf("\n");
    }
    printf("\n");
}

void cpu_mode() {
    printf("CPU mode\n");
    // Variable, timespec for time measurement
    int *MA, *MB, *MC, *ME;
    const int matrix_size = SIZE_N * SIZE_N * sizeof(int);
    int d;
    struct timespec start, end;

    // Malloc
    MA = (int*)malloc(matrix_size);
    if (MA == NULL) {
        printf("Error allocating memory for MA\n");
        return;
    }
    MB = (int*)malloc(matrix_size);
    if (MB == NULL) {
        printf("Error allocating memory for MB\n");
        free(MA);
        return;
    }
    MC = (int*)malloc(matrix_size);
    if (MC == NULL) {
        printf("Error allocating memory for MC\n");
        free(MA);
        free(MB);
        return;
    }
    ME = (int*)malloc(matrix_size);
    if (ME == NULL) {
        printf("Error allocating memory for ME\n");
        free(MA);
        free(MB);
        free(MC);
        return;
    }

    // Fill MA
    fill_matrix(MA, SIZE_N);

    // Read data from files
    read_matrix_int("data\\8192_MB_i.txt", MB, SIZE_N, SIZE_N);
    read_matrix_int("data\\8192_MC_i.txt", MC, SIZE_N, SIZE_N);
    read_matrix_int("data\\8192_ME_i.txt", ME, SIZE_N, SIZE_N);
    read_scalar_int("data\\d_i.txt", &d);

    // Start timer
    timespec_get(&start, TIME_UTC);

    // MA = MB * MC + d * ME
    i_matrix_multiply_matrix_acc(MB, MC, MA, SIZE_N);
    i_matrix_multiply_scalar_acc(ME, d, MA, SIZE_N);

    // End timer
    timespec_get(&end, TIME_UTC);

    // Show elapsed time
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.5f milliseconds\n", elapsed_time * 1000);

    // Print and save result
    print_matrix_result(MA, SIZE_N);
    write_matrix_int("result\\result_cpu_prg1.txt", MA, SIZE_N, SIZE_N);

    // Free malloc
    free(MA);
    free(MB);
    free(MC);
    free(ME);
}

void gpu_mode() {
    // Configuration checks
    if (THREADS_PER_BLOCK > 1024) {
        printf("Error: THREADS_PER_BLOCK exceeds 1024\n");
        return;
    }
    if (GRID_Y > 65535 || BLOCKS_PER_GRID > 2147483647) // "2 ^ 31 - 1" or "(1 << 31) - 1"
    {
        printf("Error: BLOCK_Y exceeds 65535 or BLOCKS_PER_GRID exceeds 2 ^ 31 - 1\n");
        return;
    }

    printf("GPU mode\n");
    // Variables
    int *MA, *MB, *MC, *ME;
    const int matrix_size = SIZE_N * SIZE_N * sizeof(int);
    int d;
    cudaError_t err;

    // CUDA events for performance measurement
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        printf("Error creating event for timing.");
        return;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        printf("Error creating event for timing.");
        return;
    }

    // Allocate memory (Host)
    err = cudaMallocHost((void**)&MA, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MA\n");
        return;
    }
    err = cudaMallocHost((void**)&MB, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MB\n");
        return;
    }
    err = cudaMallocHost((void**)&MC, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MC\n");
        return;
    }
    err = cudaMallocHost((void**)&ME, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for ME\n");
        return;
    }

    // Fill MA
    fill_matrix(MA, SIZE_N);

    // Read data from files
    read_matrix_int("data\\8192_MB_i.txt", MB, SIZE_N, SIZE_N);
    read_matrix_int("data\\8192_MC_i.txt", MC, SIZE_N, SIZE_N);
    read_matrix_int("data\\8192_ME_i.txt", ME, SIZE_N, SIZE_N);
    read_scalar_int("data\\d_i.txt", &d);

    // Device pointers
    int* d_MA = NULL;
    int* d_MB = NULL;
    int* d_MC = NULL;
    int* d_ME = NULL;
    int* d_d = NULL;

    // Allocate memory on the device
    err = cudaMalloc((void **)&d_MA, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MA on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_MB, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MB on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_MC, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MC on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_ME, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for ME on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_d, sizeof(int));
    if (err != cudaSuccess) {
        printf("Error allocating memory for d on device\n");
        return;
    }

    // Copy data to the device
    err = cudaMemcpyAsync(d_MB, MB, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying MB to the device.");
        return;
    }
    err = cudaMemcpyAsync(d_MC, MC, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying MC to the device.");
        return;
    }
    err = cudaMemcpyAsync(d_ME, ME, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying ME to the device.");
        return;
    }
    err = cudaMemcpyAsync(d_d, &d, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying d to the device.");
        return;
    }
    
    // Prepare for kernel launches
    dim3 block_dims(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_dims(GRID_X, GRID_Y, GRID_Z);

    // Launch kernels
    err = cudaEventRecord(start);
    if (err != cudaSuccess) {
        printf("Error recording start event.");
        return;
    }

    i_gpu_matrix_multiply_matrix_acc<<<grid_dims, block_dims>>>(d_MB, d_MC, d_MA, SIZE_N);
    i_gpu_matrix_multiply_scalar_acc<<<grid_dims, block_dims>>>(d_ME, d_d, d_MA, SIZE_N);

    err = cudaEventRecord(stop);
    if (err != cudaSuccess) {
        printf("Error recording stop event.");
        return;
    }
    cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Error syncronizing.");
        return;
    }

    // Copy result to the host
    err = cudaMemcpyAsync(MA, d_MA, matrix_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying result to host.");
        return;
    }

    // Free device memory
    cudaFree(d_MA);
    cudaFree(d_MB);
    cudaFree(d_MC);
    cudaFree(d_ME);

    // Show elapsed time
    float elapsed_time;
    err = cudaEventElapsedTime(&elapsed_time, start, stop);
    if (err != cudaSuccess) {
        printf("Error evaluating elapsed time.");
        return;
    }
    printf("Elapsed time: %.5f milliseconds\n", elapsed_time);
    
    // Print and save result
    print_matrix_result(MA, SIZE_N);
    write_matrix_int("result\\result_gpu_prg1.txt", MA, SIZE_N, SIZE_N);

    // Free host memory
    cudaFreeHost(MA);
    cudaFreeHost(MB);
    cudaFreeHost(MC);
    cudaFreeHost(ME);
}

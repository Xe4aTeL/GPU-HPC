/*
PRG1 - Програма 2
Задача:
A = B + MC * MD * E

Виконав:
Скоробагатько Іван ІО-13

PRG1 - Program 2
Task:
A = B + MC * MD * E

Programmed by:
Skorobagatko Ivan ІО-13
*/

#include <time.h>

#include "file_handler.h"
#include "gpu_kernel.cuh"
#include "cpu_math.h"
#include "config.h"

void fill_matrix(float* matrix, int N);
void fill_vector(float* vector, int N);
void print_vector_result(float* vector, int N);
void cpu_mode(int SIZE_N);
void gpu_mode(int SIZE_N);

int main(int argc, char* argv[]) {
    // Check if the program is run with the correct number of arguments
    if (argc != 3) {
        printf("Usage: <GPU_ENABLE> <SIZE>\n");
        exit(EXIT_FAILURE);
    }

    // Parse GPU_ENABLE from command line argument
    int GPU_ENABLE = atoi(argv[1]);

    // Parse SIZE_N from command line argument
    int SIZE_N = atoi(argv[2]);

    // Check if GPU_ENABLE is valid
    if (GPU_ENABLE != 0 && GPU_ENABLE != 1) {
        printf("Error: GPU_ENABLE must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }

    // Check if SIZE_N is valid
    if (SIZE_N < 1) {
        printf("Error: SIZE_N must be positive\n");
        exit(EXIT_FAILURE);
    }

    // Check CUBLAS_ENABLE from config.h
    if (CUBLAS_ENABLE != 0 && CUBLAS_ENABLE != 1) {
        printf("Error: CUBLAS_ENABLE must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }

    // Create timer for measuring execution time
    struct timespec start, end; 

    // Start the timer
    timespec_get(&start, TIME_UTC);

    // CPU mode memory allocation
    if (GPU_ENABLE == 0)
        cpu_mode(SIZE_N);

    // GPU mode memory allocation
    if (GPU_ENABLE == 1)
        gpu_mode(SIZE_N);

    // Stop the timer
    timespec_get(&end, TIME_UTC);

    // Show elapsed time
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time executing: %.5f milliseconds\n", elapsed_time * 1000);

    printf("Done\n");
    exit(EXIT_SUCCESS);
}

void fill_matrix(float* matrix, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i * N + j] = 0.0f;
}

void fill_vector(float* vector, int N) {
    for (int i = 0; i < N; i++)
        vector[i] = 0.0f;
}

void print_vector_result(float* vector, int N) {
    if (N > 32) {
        printf("Result is too large to print.\n");
        return;
    }

    printf("Result: \n");
    for (int i = 0; i < N; i++)
        printf("%f ", vector[i]);

    printf("\n");
}

void cpu_mode(int SIZE_N) {
    printf("CPU mode\n");
    // Variables
    int matrix_size = SIZE_N * SIZE_N * sizeof(float);
    int vector_size = SIZE_N * sizeof(float);
    float *A, *B, *MC, *MD, *E, *MX;
    struct timespec start, end;

    // Malloc
    A = (float*)malloc(vector_size);
    if (A == NULL) {
        printf("Error allocating memory for A\n");
        exit(EXIT_FAILURE);
    }
    B = (float*)malloc(vector_size);
    if (B == NULL) {
        printf("Error allocating memory for B\n");
        free(A);
        exit(EXIT_FAILURE);
    }
    MC = (float*)malloc(matrix_size);
    if (MC == NULL) {
        printf("Error allocating memory for MC\n");
        free(A);
        free(B);
        exit(EXIT_FAILURE);
    }
    MD = (float*)malloc(matrix_size);
    if (MD == NULL) {
        printf("Error allocating memory for MD\n");
        free(A);
        free(B);
        free(MC);
        exit(EXIT_FAILURE);
    }
    E = (float*)malloc(vector_size);
    if (E == NULL) {
        printf("Error allocating memory for E\n");
        free(A);
        free(B);
        free(MC);
        free(MD);
        exit(EXIT_FAILURE);
    }
    MX = (float*)malloc(matrix_size);
    if (MX == NULL) {
        printf("Error allocating memory for MX\n");
        free(A);
        free(B);
        free(MC);
        free(MD);
        free(E);
        exit(EXIT_FAILURE);
    }

    // Fill A, MX
    fill_vector(A, SIZE_N);
    fill_matrix(MX, SIZE_N);

    // Read data from files
    switch (SIZE_N) {
        case 256:
            read_vector_float("data\\256_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\256_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\256_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\256_E_f.txt", E, SIZE_N);
            break;
        case 512:
            read_vector_float("data\\512_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\512_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\512_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\512_E_f.txt", E, SIZE_N);
            break;
        case 1024:
            read_vector_float("data\\1024_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\1024_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\1024_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\1024_E_f.txt", E, SIZE_N);
            break;
        case 2048:
            read_vector_float("data\\2048_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\2048_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\2048_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\2048_E_f.txt", E, SIZE_N);
            break;
        case 4096:
            read_vector_float("data\\4096_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\4096_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\4096_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\4096_E_f.txt", E, SIZE_N);
            break;
        case 8192:
            read_vector_float("data\\8192_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\8192_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\8192_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\8192_E_f.txt", E, SIZE_N);
            break;
        case 16384:
            read_vector_float("data\\16384_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\16384_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\16384_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\16384_E_f.txt", E, SIZE_N);
            break;
        default:
            printf("Unusual size of input data, reading B_f.txt, MC_f.txt, MD_f.txt, E_f.txt\n");
            read_vector_float("data\\B_f.txt", B, SIZE_N);
            read_matrix_float("data\\MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\E_f.txt", E, SIZE_N);
            break;
    }

    // Start timer
    timespec_get(&start, TIME_UTC);

    // A = B + MC * MD * E
    f_matrix_multiply_matrix_acc(MC, MD, MX, SIZE_N);
    f_matrix_multiply_vector_acc(MX, E, A, SIZE_N);
    f_vector_add_vector(A, B, SIZE_N);

    // End timer
    timespec_get(&end, TIME_UTC);

    // Print and save result
    print_vector_result(A, SIZE_N);
    switch(SIZE_N)
    {
        case 256:
            write_vector_float("result\\256_cpu_prg2.txt", A, SIZE_N);
            break;
        case 512:
            write_vector_float("result\\512_cpu_prg2.txt", A, SIZE_N);
            break;
        case 1024:
            write_vector_float("result\\1024_cpu_prg2.txt", A, SIZE_N);
            break;
        case 2048:
            write_vector_float("result\\2048_cpu_prg2.txt", A, SIZE_N);
            break;
        case 4096:
            write_vector_float("result\\4096_cpu_prg2.txt", A, SIZE_N);
            break;
        case 8192:
            write_vector_float("result\\8192_cpu_prg2.txt", A, SIZE_N);
            break;
        case 16384:
            write_vector_float("result\\16384_cpu_prg2.txt", A, SIZE_N);
            break;
        default:
            write_vector_float("result\\result_cpu_prg2.txt", A, SIZE_N);
            break;
    }

    // Show elapsed time
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time calculating: %.5f milliseconds\n", elapsed_time * 1000);

    // Free malloc
    free(A);
    free(B);
    free(MC);
    free(MD);
    free(E);
    free(MX);
}

void gpu_mode(int SIZE_N) {
    // Configuration checks
    if (THREADS_PER_BLOCK > 1024) {
        printf("Error: THREADS_PER_BLOCK exceeds 1024\n");
        return;
    }
    if (GRID_Y(BLOCK_Y) > 65535 || BLOCKS_PER_GRID(SIZE_N) > 2147483647) // "2 ^ 31 - 1" or "(1 << 31) - 1"
    {
        printf("Error: BLOCK_Y exceeds 65535 or BLOCKS_PER_GRID exceeds 2 ^ 31 - 1\n");
        return;
    }

    printf("GPU mode\n");
    // Variables
    int matrix_size = SIZE_N * SIZE_N * sizeof(float);
    int vector_size = SIZE_N * sizeof(float);
    float *A, *B, *MC, *MD, *E, *MX;
    cudaError_t err;
    cudaEvent_t start, stop; // CUDA events for performance measurement

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        printf("Error creating event for timing.");
        exit(EXIT_FAILURE);
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        printf("Error creating event for timing.");
        exit(EXIT_FAILURE);
    }

    // Allocate memory (Host)
    err = cudaMallocHost((void**)&A, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for A\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMallocHost((void**)&B, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for B\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMallocHost((void**)&MC, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MC\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMallocHost((void**)&MD, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MD\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMallocHost((void**)&E, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for E\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMallocHost((void**)&MX, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MX\n");
        exit(EXIT_FAILURE);
    }

    // Fill A, MX
    fill_vector(A, SIZE_N);
    fill_matrix(MX, SIZE_N);

    // Read data from files
    switch (SIZE_N) {
        case 256:
            read_vector_float("data\\256_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\256_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\256_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\256_E_f.txt", E, SIZE_N);
            break;
        case 512:
            read_vector_float("data\\512_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\512_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\512_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\512_E_f.txt", E, SIZE_N);
            break;
        case 1024:
            read_vector_float("data\\1024_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\1024_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\1024_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\1024_E_f.txt", E, SIZE_N);
            break;
        case 2048:
            read_vector_float("data\\2048_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\2048_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\2048_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\2048_E_f.txt", E, SIZE_N);
            break;
        case 4096:
            read_vector_float("data\\4096_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\4096_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\4096_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\4096_E_f.txt", E, SIZE_N);
            break;
        case 8192:
            read_vector_float("data\\8192_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\8192_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\8192_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\8192_E_f.txt", E, SIZE_N);
            break;
        case 16384:
            read_vector_float("data\\16384_B_f.txt", B, SIZE_N);
            read_matrix_float("data\\16384_MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\16384_MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\16384_E_f.txt", E, SIZE_N);
            break;
        default:
            printf("Unusual size of input data, reading B_f.txt, MC_f.txt, MD_f.txt, E_f.txt\n");
            read_vector_float("data\\B_f.txt", B, SIZE_N);
            read_matrix_float("data\\MC_f.txt", MC, SIZE_N, SIZE_N);
            read_matrix_float("data\\MD_f.txt", MD, SIZE_N, SIZE_N);
            read_vector_float("data\\E_f.txt", E, SIZE_N);
            break;
    }

    // Device pointers
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_MC = NULL;
    float* d_MD = NULL;
    float* d_E = NULL;
    float* d_MX = NULL;

    // Allocate memory on the device
    err = cudaMalloc((void **)&d_A, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for A on device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_B, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for B on device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_MC, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MC on device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_MD, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MD on device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_E, vector_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for E on device\n");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&d_MX, matrix_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory for MX on device\n");
        exit(EXIT_FAILURE);
    }

    // Prepare for kernel launches
    dim3 block_dims(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid_dims(GRID_X(BLOCK_X), GRID_Y(BLOCK_Y), GRID_Z);
    // For Efficient vector addition
    int vec_block_dims = 32;
    int vec_grid_dims = (SIZE_N + vec_block_dims - 1) / vec_block_dims;

    if (CUBLAS_ENABLE == 0) {
        printf("GPU - Using kernel calls.\n");

        // Copy data to the device
        err = cudaMemcpyAsync(d_B, B, vector_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying B to the device.");
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpyAsync(d_MC, MC, matrix_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying MC to the device.");
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpyAsync(d_MD, MD, matrix_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying MD to the device.");
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpyAsync(d_E, E, vector_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying E to the device.");
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpyAsync(d_MX, MX, matrix_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying MX to the device.");
            exit(EXIT_FAILURE);
        }

        // Launch kernels
        err = cudaEventRecord(start);
        if (err != cudaSuccess) {
            printf("Error recording start event.");
            exit(EXIT_FAILURE);
        }

        f_gpu_matrix_multiply_matrix_acc<<<grid_dims, block_dims>>>(d_MC, d_MD, d_MX, SIZE_N);
        f_gpu_matrix_multiply_vector_acc<<<grid_dims, block_dims>>>(d_MX, d_E, d_A, SIZE_N);
        f_gpu_vector_add_vector<<<vec_grid_dims, vec_block_dims>>>(d_A, d_B, SIZE_N);

        err = cudaEventRecord(stop);
        if (err != cudaSuccess) {
            printf("Error recording stop event.");
            exit(EXIT_FAILURE);
        }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            printf("Error syncronizing.");
            exit(EXIT_FAILURE);
        }

        // Copy result to the host
        err = cudaMemcpy(A, d_A, vector_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying result to host.");
            exit(EXIT_FAILURE);
        }
    }

    if (CUBLAS_ENABLE == 1) {
        printf("GPU - Using cuBLAS calls.\n");

        // Variables for cuBLAS
        cublasHandle_t h_cublas;
        cublasStatus_t cublas_status;
        float alpha = 1.0f;
        float beta = 1.0f;

        // Init cuBLAS
        cublas_status = cublasCreate_v2(&h_cublas);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error starting cuBLAS.");
            exit(EXIT_FAILURE);
        }
        cublas_status = cublasSetMathMode(h_cublas, CUBLAS_DEFAULT_MATH);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error setting math mode cuBLAS.");
            exit(EXIT_FAILURE);
        }

        // Copy data to the device
        cublas_status = cublasSetVectorAsync(SIZE_N, sizeof(float), B, 1, d_B, 1, 0);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying B to the device.");
            exit(EXIT_FAILURE);
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), MC, SIZE_N, d_MC, SIZE_N, 0);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying MC to the device.");
            exit(EXIT_FAILURE);
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), MD, SIZE_N, d_MD, SIZE_N, 0);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying MD to the device.");
            exit(EXIT_FAILURE);
        }
        cublas_status = cublasSetVectorAsync(SIZE_N, sizeof(float), E, 1, d_E, 1, 0);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying E to the device.");
            exit(EXIT_FAILURE);
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), MX, SIZE_N, d_MX, SIZE_N, 0);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying MX to the device.");
            exit(EXIT_FAILURE);
        }

        // Start the timer
        err = cudaEventRecord(start);
        if (err != cudaSuccess) {
            printf("Error recording start event.");
            exit(EXIT_FAILURE);
        }

        // Perform calculation
        // MX = alpha (MC * MD) + beta * MX, S for single-precision
        cublas_status = cublasSgemm_v2(h_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE_N, SIZE_N, SIZE_N, &alpha, d_MC, SIZE_N, d_MD, SIZE_N, &beta, d_MX, SIZE_N);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error performing MX = MC * MD.");
            exit(EXIT_FAILURE);
        }
        // A = alpha (MX * E) + beta * A, S for single-precision
        cublas_status = cublasSgemv_v2(h_cublas, CUBLAS_OP_N, SIZE_N, SIZE_N, &alpha, d_MX, SIZE_N, d_E, 1, &beta, d_A, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error performing A = MX * E.");
            exit(EXIT_FAILURE);
        }
        // A += B, cuBLAS don't provide vector addition
        f_gpu_vector_add_vector<<<vec_grid_dims, vec_block_dims>>>(d_A, d_B, SIZE_N);

        err = cudaEventRecord(stop);
        if (err != cudaSuccess) {
            printf("Error recording stop event.");
            exit(EXIT_FAILURE);
        }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            printf("Error syncronizing.");
            exit(EXIT_FAILURE);
        }

        // Copy result to the host
        cublas_status = cublasGetVector(SIZE_N, sizeof(float), d_A, 1, A, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error copying result to host.");
            exit(EXIT_FAILURE);
        }
        
        // Destroy cuBLAS
        cublas_status = cublasDestroy_v2(h_cublas);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error destroying cublas.");
            exit(EXIT_FAILURE);
        }
    }

    // Sync the device
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error syncronizing.");
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_MC);
    cudaFree(d_MD);
    cudaFree(d_E);
    cudaFree(d_MX);
    
    // Print and save result
    print_vector_result(A, SIZE_N);
    switch(SIZE_N)
    {
        case 256:
            write_vector_float("result\\256_gpu_prg2.txt", A, SIZE_N);
            break;
        case 512:
            write_vector_float("result\\512_gpu_prg2.txt", A, SIZE_N);
            break;
        case 1024:
            write_vector_float("result\\1024_gpu_prg2.txt", A, SIZE_N);
            break;
        case 2048:
            write_vector_float("result\\2048_gpu_prg2.txt", A, SIZE_N);
            break;
        case 4096:
            write_vector_float("result\\4096_gpu_prg2.txt", A, SIZE_N);
            break;
        case 8192:
            write_vector_float("result\\8192_gpu_prg2.txt", A, SIZE_N);
            break;
        case 16384:
            write_vector_float("result\\16384_gpu_prg2.txt", A, SIZE_N);
            break;
        default:
            write_vector_float("result\\result_gpu_prg2.txt", A, SIZE_N);
            break;
    }

    // Calculate elapsed time
    float elapsed_time;
    err = cudaEventElapsedTime(&elapsed_time, start, stop);
    if (err != cudaSuccess) {
        printf("Error evaluating elapsed time.");
        exit(EXIT_FAILURE);
    }
    printf("Elapsed time calculating: %.5f milliseconds\n", elapsed_time);

    // Free host memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(MC);
    cudaFreeHost(MD);
    cudaFreeHost(E);
    cudaFreeHost(MX);
}

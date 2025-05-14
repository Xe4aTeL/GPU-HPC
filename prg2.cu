#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpu_kernel.cuh"
#include "cpu_math.h"
#include "file_handler.h"
#include "config.h"

void fill_matrix(float* matrix, int N);
void fill_vector(float* vector, int N);
void print_matrix_result(float* matrix, int N);
void print_vector_result(float* vector, int N);
void cpu_mode();
void gpu_mode();

int main(int argc, char* argv[])
{
    // Check if the program is run with the correct number of arguments
    if (argc != 2)
    {
        printf("Usage: %s <GPU_ENABLE>\n", argv[0]);
        return -1;
    }

    // Parse GPU_ENABLE from command line argument
    int GPU_ENABLE = atoi(argv[1]);

    // Check if GPU_ENABLE is valid
    if (GPU_ENABLE != 0 && GPU_ENABLE != 1)
    {
        printf("Error: GPU_ENABLE must be 0 or 1\n");
        return -1;
    }

    // Check CUBLAS_ENABLE from config.h
    if (CUBLAS_ENABLE != 0 && CUBLAS_ENABLE != 1)
    {
        printf("Error: CUBLAS_ENABLE must be 0 or 1\n");
        return -1;
    }

    // CPU mode memory allocation
    if (GPU_ENABLE == 0)
    {
        cpu_mode();
    }

    // GPU mode memory allocation
    if (GPU_ENABLE == 1)
    {
        gpu_mode();
    }

    printf("Done\n");
    return 0;

    return 0;
}

void fill_matrix(float* matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = 0.0f;
        }
    }
}

void fill_vector(float* vector, int N)
{
    for (int i = 0; i < N; i++)
    {
        vector[i] = 0.0f;
    }
}

void print_matrix_result(float* matrix, int N)
{
    if (N > 32)
    {
        printf("Result is too large to print.\n");
        return;
    }

    printf("Result: ");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }

    printf("\n");
}

void print_vector_result(float* vector, int N)
{
    if (N > 32)
    {
        printf("Result is too large to print.\n");
        return;
    }

    printf("Result: ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", vector[i]);
    }

    printf("\n");
}

void cpu_mode()
{
    printf("CPU mode\n");
    // Variables
    int matrix_size = SIZE_N * SIZE_N * sizeof(float);
    int vector_size = SIZE_N * sizeof(float);
    float *A, *B, *MC, *MD, *E, *Mtmp;

    // Malloc
    A = (float*)malloc(vector_size);
    if (A == NULL)
    {
        printf("Error allocating memory for A\n");
        return;
    }
    B = (float*)malloc(vector_size);
    if (B == NULL)
    {
        printf("Error allocating memory for B\n");
        free(A);
        return;
    }
    MC = (float*)malloc(matrix_size);
    if (MC == NULL)
    {
        printf("Error allocating memory for MC\n");
        free(A);
        free(B);
        return;
    }
    MD = (float*)malloc(matrix_size);
    if (MD == NULL)
    {
        printf("Error allocating memory for MD\n");
        free(A);
        free(B);
        free(MC);
        return;
    }
    E = (float*)malloc(vector_size);
    if (E == NULL)
    {
        printf("Error allocating memory for E\n");
        free(A);
        free(B);
        free(MC);
        free(MD);
        return;
    }
    Mtmp = (float*)malloc(matrix_size);
    if (Mtmp == NULL)
    {
        printf("Error allocating memory for Mtmp\n");
        free(A);
        free(B);
        free(MC);
        free(MD);
        free(E);
        return;
    }


    // Fill A, Mtmp
    fill_vector(A, SIZE_N);
    fill_matrix(Mtmp, SIZE_N);

    // Read data from files
    read_vector_float("data\\256_B_f.txt", B, SIZE_N);
    read_matrix_float("data\\256_MC_f.txt", MC, SIZE_N, SIZE_N);
    read_matrix_float("data\\256_MD_f.txt", MD, SIZE_N, SIZE_N);
    read_vector_float("data\\256_E_f.txt", E, SIZE_N);

    // Timer + Start
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);

    // A = B + MC * MD * E
    f_matrix_multiply_matrix_acc(MC, MD, Mtmp, SIZE_N);
    f_matrix_multiply_vector_acc(Mtmp, E, A, SIZE_N);
    f_vector_add_vector(A, B, SIZE_N);

    // End timer
    timespec_get(&end, TIME_UTC);

    // Show elapsed time
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.5f milliseconds\n", elapsed_time * 1000);

    // Print and save result
    print_vector_result(A, SIZE_N);
    write_vector_float("result\\result_cpu_prg2.txt", A, SIZE_N);

    // Free malloc
    free(A);
    free(B);
    free(MC);
    free(MD);
    free(E);
    free(Mtmp);
    A = NULL;
    B = NULL;
    MC = NULL;
    MD = NULL;
    E = NULL;
    Mtmp = NULL;

    return;
}

void gpu_mode()
{
    if (THREADS_PER_BLOCK > 1024)
    {
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
    int matrix_size = SIZE_N * SIZE_N * sizeof(float);
    int vector_size = SIZE_N * sizeof(float);
    float *A, *B, *MC, *MD, *E, *Mtmp;
    cudaError_t err;

    // CUDA events for performance measurement
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        printf("Error creating event for timing.");
        return;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess)
    {
        printf("Error creating event for timing.");
        return;
    }

    cudaStream_t stream;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("Error creating stream.");
        return;
    }

    // Allocate memory (Host)
    err = cudaMallocHost((void**)&A, vector_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for A\n");
        return;
    }
    err = cudaMallocHost((void**)&B, vector_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for B\n");
        cudaFreeHost(A);
        return;
    }
    err = cudaMallocHost((void**)&MC, matrix_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for MC\n");
        cudaFreeHost(A);
        cudaFreeHost(B);
        return;
    }
    err = cudaMallocHost((void**)&MD, matrix_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for MD\n");
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        return;
    }
    err = cudaMallocHost((void**)&E, vector_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for E\n");
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        return;
    }
    err = cudaMallocHost((void**)&Mtmp, matrix_size);
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for Mtmp\n");
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        return;
    }

    // Fill A, Mtmp
    fill_vector(A, SIZE_N);
    fill_matrix(Mtmp, SIZE_N);

    // Read data from files
    read_vector_float("data\\256_B_f.txt", B, SIZE_N);
    read_matrix_float("data\\256_MC_f.txt", MC, SIZE_N, SIZE_N);
    read_matrix_float("data\\256_MD_f.txt", MD, SIZE_N, SIZE_N);
    read_vector_float("data\\256_E_f.txt", E, SIZE_N);

    // Device pointers
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_MC = NULL;
    float* d_MD = NULL;
    float* d_E = NULL;
    float* d_Mtmp = NULL;

    // Allocate memory on the device
    err = cudaMalloc((void **)&d_A, vector_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        printf("Error allocating memory for A on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_B, vector_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        cudaFree(d_A);
        printf("Error allocating memory for B on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_MC, matrix_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        cudaFree(d_A);
        cudaFree(d_B);
        printf("Error allocating memory for MC on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_MD, matrix_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        printf("Error allocating memory for MD on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_E, vector_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        printf("Error allocating memory for MD on device\n");
        return;
    }
    err = cudaMalloc((void **)&d_Mtmp, matrix_size);
    if (err != cudaSuccess)
    {
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        cudaFree(d_E);
        printf("Error allocating memory for Mtmp on device\n");
        return;
    }

    // Prepare for kernel launches
    int block_x = BLOCK_X;
    int block_y = BLOCK_Y;
    int block_z = BLOCK_Z;
    int grid_x = GRID_X;
    int grid_y = GRID_Y;
    int grid_z = GRID_Z;
    dim3 block_dims(block_x, block_y, block_z);
    dim3 grid_dims(grid_x, grid_y, grid_z);

    if (CUBLAS_ENABLE == 0)
    {
        printf("GPU - Using kernel calls.\n");
        // Copy data to the device
        err = cudaMemcpyAsync(d_B, B, vector_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error copying B to the device.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaMemcpyAsync(d_MC, MC, matrix_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error copying MC to the device.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaMemcpyAsync(d_MD, MD, matrix_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error copying MD to the device.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaMemcpyAsync(d_E, E, vector_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error copying E to the device.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaMemcpyAsync(d_Mtmp, Mtmp, matrix_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            printf("Error copying Mtmp to the device.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Launch kernels
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess)
    {
        printf("Error recording start event.");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        cudaFree(d_E);
        cudaFree(d_Mtmp);
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        return;
    }
        f_gpu_matrix_multiply_matrix_acc<<<grid_dims, block_dims, 0, stream>>>(d_MC, d_MD, d_Mtmp, SIZE_N);
        f_gpu_matrix_multiply_vector_acc<<<grid_dims, block_dims, 0, stream>>>(d_Mtmp, d_E, d_A, SIZE_N);
        f_gpu_vector_add_vector<<<grid_dims, block_dims, 0, stream>>>(d_A, d_B, SIZE_N);

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
    {
        printf("Error syncronizing stream after kernels.");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        cudaFree(d_E);
        cudaFree(d_Mtmp);
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        return;
    }
        err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess)
    {
        printf("Error recording stop event.");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        cudaFree(d_E);
        cudaFree(d_Mtmp);
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        return;
    }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess)
    {
        printf("Error recording stop event.");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_MC);
        cudaFree(d_MD);
        cudaFree(d_E);
        cudaFree(d_Mtmp);
        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(MC);
        cudaFreeHost(MD);
        cudaFreeHost(E);
        cudaFreeHost(Mtmp);
        return;
    }

        // Copy result to the host
        err = cudaMemcpy(A, d_A, vector_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            printf("Error copying result to host.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
    }

    if (CUBLAS_ENABLE == 1)
    {
        printf("GPU - Using cuBLAS calls.\n");
        // Variables for cuBLAS
        cublasHandle_t h_cublas;
        cublasStatus_t cublas_status;
        float alpha = 1.0f;
        float beta = 0.0f;

        // Init cuBLAS
        cublas_status = cublasCreate_v2(&h_cublas);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Error starting cuBLAS.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Set stream
        cublas_status = cublasSetStream_v2(h_cublas, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to set a stream.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Copy data to the device
        cublas_status = cublasSetVectorAsync(SIZE_N, sizeof(float), B, 1, d_B, 1, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy vector B.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), MC, SIZE_N, d_MC, SIZE_N, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy matrix MC.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), MD, SIZE_N, d_MD, SIZE_N, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy matrix MD.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        cublas_status = cublasSetVectorAsync(SIZE_N, sizeof(float), E, 1, d_E, 1, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy vector E.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        cublas_status = cublasSetMatrixAsync(SIZE_N, SIZE_N, sizeof(float), Mtmp, SIZE_N, d_Mtmp, SIZE_N, stream);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy matrix Mtmp.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Start the timer
        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess)
        {
            printf("Error starting the timer.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Perform calculation
        cublas_status = cublasSgemm_v2( // MC = alpha (MA * MB) + beta * MC, S for single-precision
            h_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N, // Don't transpose A and B
            SIZE_N, SIZE_N, SIZE_N, // Sizes of matrices
            &alpha,
            d_MC, SIZE_N, // MC
            d_MD, SIZE_N, // MD
            &beta,
            d_Mtmp, SIZE_N // Mtmp
        );
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Error performing MC * MD.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        cublas_status = cublasSgemv_v2( // A = alpha (Mtmp * E) + beta * A, S for single-precision
            h_cublas,
            CUBLAS_OP_N,
            SIZE_N, SIZE_N,
            &alpha,
            d_Mtmp, SIZE_N, // MC * MD
            d_E, 1, // E
            &beta,
            d_A, 1 // A
        );

        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Error performing Mtmp * E.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        f_gpu_vector_add_vector<<<grid_dims, block_dims, 0, stream>>>(d_A, d_B, SIZE_N);

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            printf("Error syncronizing stream.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess)
        {
            printf("Error stopping the timer.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess)
        {
            printf("Error syncronizing the timer.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }

        // Copy result to the host
        cublas_status = cublasGetVector(SIZE_N, sizeof(float), d_A, 1, A, 1);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Failed to copy vector A.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
        
        // Destroy cuBLAS
        cublas_status = cublasDestroy_v2(h_cublas);
        if (cublas_status != CUBLAS_STATUS_SUCCESS)
        {
            printf("Error destroying cublas.");
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_MC);
            cudaFree(d_MD);
            cudaFree(d_E);
            cudaFree(d_Mtmp);
            cudaFreeHost(A);
            cudaFreeHost(B);
            cudaFreeHost(MC);
            cudaFreeHost(MD);
            cudaFreeHost(E);
            cudaFreeHost(Mtmp);
            return;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_MC);
    cudaFree(d_MD);
    cudaFree(d_E);
    cudaFree(d_Mtmp);

    // Calculate elapsed time
    float elapsed_time;
    err = cudaEventElapsedTime(&elapsed_time, start, stop);
    if (err != cudaSuccess)
    {
        printf("Error evaluating elapsed time.");
        return;
    }
    printf("Elapsed time: %.5f milliseconds\n", elapsed_time);
    
    // Print result
    print_vector_result(A, SIZE_N);
    // Write result to file
    write_vector_float("result\\result_gpu_prg2.txt", A, SIZE_N);

    // Free host memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(MC);
    cudaFreeHost(MD);
    cudaFreeHost(E);
    cudaFreeHost(Mtmp);

    return;
}

#include "gpu_kernel.cuh"

// PRG1
__global__ void i_gpu_matrix_multiply_matrix_acc(int* matrix1, int* matrix2, int* matrix_res, int N)
{
    int idx = blockDim.y * blockIdx.y + threadIdx.y;
    int idy = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N && idy < N)
    {
        int sum = 0;
        
        #pragma unroll
        for (int k = 0; k < N; k++)
        {
            sum += matrix1[idx * N + k] * matrix2[k * N + idy];
        }
        matrix_res[idx * N + idy] = sum;
    }
}

__global__ void i_gpu_matrix_multiply_scalar_acc(int* matrix, int* scalar, int* matrix_res, int N)
{
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idy = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N && idy < N)
        matrix_res[idx * N + idy] += matrix[idx * N + idy] * *(scalar);
}

// PRG2
__global__ void f_gpu_matrix_multiply_matrix_acc(float* matrix1, float* matrix2, float* matrix_res, int N)
{
    int idx = blockDim.y * blockIdx.y + threadIdx.y;
    int idy = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N && idy < N)
    {
        float sum = 0;
        
        #pragma unroll
        for (int k = 0; k < N; k++)
        {
            sum += matrix1[idx * N + k] * matrix2[k * N + idy];
        }
        matrix_res[idx * N + idy] = sum;
    }
}

__global__ void f_gpu_matrix_multiply_vector_acc(float* matrix, float* vector, float* vector_res, int N)
{
    int idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx < N)
    {
        float sum = 0;

        #pragma unroll
        for (int k = 0; k < N; k++)
        {
            sum += matrix[idx * N + k] * vector[k];
        }
        vector_res[idx] = sum;
    }
}

__global__ void f_gpu_vector_add_vector(float* vector1, float* vector2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N)
        vector1[idx] += vector2[idx];
}

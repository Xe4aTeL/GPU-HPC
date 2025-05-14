#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#if __cplusplus
extern "C" {
#endif

__global__ void i_gpu_matrix_multiply_matrix_acc(int* matrix1, int* matrix2, int* matrix_res, int N);
__global__ void i_gpu_matrix_multiply_vector_acc(int* matrix, int* vector, int* vector_res, int N);
__global__ void i_gpu_matrix_multiply_scalar_acc(int* matrix, int* scalar, int* matrix_res, int N);
__global__ void i_gpu_vector_add_vector(int* vector1, int* vector2, int N);
__global__ void f_gpu_matrix_multiply_matrix_acc(float* matrix1, float* matrix2, float* matrix_res, int N);
__global__ void f_gpu_matrix_multiply_vector_acc(float* matrix, float* vector, float* vector_res, int N);
__global__ void f_gpu_matrix_multiply_scalar_acc(float* matrix, float* scalar, float* matrix_res, int N);
__global__ void f_gpu_vector_add_vector(float* vector1, float* vector2, int N);

#if __cplusplus
}
#endif

#endif
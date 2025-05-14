#include "cpu_math.h"

void i_matrix_multiply_matrix_acc(int* matrix1, int* matrix2, int* matrix_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                matrix_res[i * N + j] += matrix1[i * N + k] * matrix2[k * N + j];
            }
        }
    }
}

void i_matrix_multiply_vector_acc(int* matrix, int* vector, int* vector_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            vector_res[i] += matrix[i * N + j] * vector[j];
        }
    }
}

void i_matrix_multiply_scalar_acc(int* matrix, int scalar, int* matrix_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix_res[i * N + j] += matrix[i * N + j] * scalar;
        }
    }
}

void i_vector_add_vector(int* vector1, int* vector2, int N)
{
    for (int i = 0; i < N; i++)
    {
        vector1[i] += vector2[i];
    }
}

void f_matrix_multiply_matrix_acc(float* matrix1, float* matrix2, float* matrix_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                matrix_res[i * N + j] += matrix1[i * N + k] * matrix2[k * N + j];
            }
        }
    }
}

void f_matrix_multiply_vector_acc(float* matrix, float* vector, float* vector_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            vector_res[i] += matrix[i * N + j] * vector[j];
        }
    }
}

void f_matrix_multiply_scalar_acc(float* matrix, float scalar, float* matrix_res, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix_res[i * N + j] += matrix[i * N + j] * scalar;
        }
    }
}

void f_vector_add_vector(float* vector1, float* vector2, int N)
{
    for (int i = 0; i < N; i++)
    {
        vector1[i] += vector2[i];
    }
}

#ifndef CPU_MATH_H
#define CPU_MATH_H

#if __cplusplus
extern "C" {
#endif

// PRG1
void i_matrix_multiply_matrix_acc(int* matrix1, int* matrix2, int* matrix_res, int N);
void i_matrix_multiply_scalar_acc(int* matrix, int scalar, int* matrix_res, int N);
// PRG2
void f_matrix_multiply_matrix_acc(float* matrix1, float* matrix2, float* matrix_res, int N);
void f_matrix_multiply_vector_acc(float* matrix, float* vector, float* vector_res, int N);
void f_vector_add_vector(float* vector1, float* vector2, int N);

#if __cplusplus
}
#endif

#endif // CPU_MATH_H
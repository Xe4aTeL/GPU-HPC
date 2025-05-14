#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include <stdio.h>
#include <stdbool.h>

#if __cplusplus
extern "C" {
#endif

void read_matrix_float(const char* filename, float* matrix, int N, int M);
void read_matrix_int(const char* filename, int* matrix, int N, int M);
void read_vector_float(const char* filename, float* vector, int N);
void read_vector_int(const char* filename, int* vector, int N);
void read_scalar_float(const char* filename, float* scalar);
void read_scalar_int(const char* filename, int* scalar);
void write_matrix_float(const char* filename, float* matrix, int N, int M);
void write_matrix_int(const char* filename, int* matrix, int N, int M);
void write_vector_float(const char* filename, float* vector, int N);
void write_vector_int(const char* filename, int* vector, int N);

#if __cplusplus
}
#endif

#endif // FILE_HANDLER_H
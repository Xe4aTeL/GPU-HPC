#include "file_handler.h"

// PRG1
void read_matrix_int(const char* filename, int* matrix, int N, int M) {
    FILE* file = fopen(filename, "r");

    if (file == NULL)  {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (fscanf(file, "%d ", &matrix[i * N + j]) != 1) {
                printf("Error reading data from file: %s\n", filename);
                fclose(file);
                return;
            }
        }
    }

    fclose(file);
    printf("Data read from file: %s\n", filename);
}

void read_scalar_int(const char* filename, int* scalar) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    if (fscanf(file, "%d", scalar) != 1) {
        printf("Error reading data from file: %s\n", filename);
        fclose(file);
        return;
    }
    
    fclose(file);
    printf("Data read from file: %s\n", filename);
}

void write_matrix_int(const char* filename, int* metrix, int N, int M) {
    FILE* file = fopen(filename, "w");

    if (file == NULL)  {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            fprintf(file, "%d ", metrix[i * N + j]);
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Data written to file: %s\n", filename);
}

// PRG2
void read_matrix_float(const char* filename, float* matrix, int N, int M) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            if (fscanf(file, "%f ", &matrix[i * N + j]) != 1) {
                printf("Error reading data from file: %s\n", filename);
                fclose(file);
                return;
            }

    fclose(file);
    printf("Data read from file: %s\n", filename);
}

void read_vector_float(const char* filename, float* vector, int N) {
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++)
        if (fscanf(file, "%f ", &vector[i]) != 1) {
            printf("Error reading data from file: %s\n", filename);
            fclose(file);
            return;
        }

    fclose(file);
    printf("Data read from file: %s\n", filename);
}

void write_vector_float(const char* filename, float* vector, int N) {
    FILE* file = fopen(filename, "w");

    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++)
        fprintf(file, "%f ", vector[i]);

    fclose(file);
    printf("Data written to a file: %s\n", filename);
}

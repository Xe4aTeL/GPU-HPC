#pragma once

#include <math.h>

// CUDA Configuration
// Block (x * y * z) = number of threads
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_Z 1
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y * BLOCK_Z) // shouldn't exceed 1024 for CUDA compute capability 5.x and above

// Grid (x * y * z) = number of blocks
#define GRID_X(SIZE_N) ((int) ceil((float) SIZE_N / BLOCK_X))
#define GRID_Y(SIZE_N) ((int) ceil((float) SIZE_N / BLOCK_Y))     // shouldn't exceed 65535 for CUDA compute capability 5.x and above
#define GRID_Z 1                                                                    // shouldn't exceed 65535 for CUDA compute capability 5.x and above
#define BLOCKS_PER_GRID(SIZE_N) (GRID_X(SIZE_N) * GRID_Y(SIZE_N) * GRID_Z)          // shouldn't exceed 2^31 - 1 for CUDA compute capability 5.x and above

// Use cuBLAS
#define CUBLAS_ENABLE 0

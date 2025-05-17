NVCC := nvcc
NVCC_FLAGS := -Wno-deprecated-gpu-targets -lineinfo -lcublas
NVCC_DEPENDENCIES_FLAGS := -c
NVCC_DEBUG_FLAGS := -Wno-deprecated-gpu-targets -g -G -lcublas

# Change this to either prg1 or prg2
TARGET := prg1
OBJ := file_handler.obj cpu_math.obj gpu_kernel.obj
GPU_TARGET_MODULE := gpu_kernel.cu

all: release

release: ${OBJ}
	${NVCC} ${NVCC_FLAGS} -o ${TARGET}.exe $^ ${TARGET}.cu

debug: ${OBJ}
	${NVCC} ${NVCC_DEBUG_FLAGS} -o ${TARGET}_debug.exe $^ ${TARGET}.cu

%.obj: %.c
	${NVCC} ${NVCC_DEPENDENCIES_FLAGS} $^ -o $@

%.obj: %.cu
	${NVCC} ${NVCC_DEPENDENCIES_FLAGS} $^ -o $@

clean:
	del *.exp *.lib *.exe, *.pdb *.obj *.ilk result\\*.txt
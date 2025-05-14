# HPC software using GPU
Made for bachelor's degree project

Developed by: Skorobagatko Ivan IO-13


---
## Features
- Supports both CPU and GPU execution of result evaluation
- Contains 2 different programs
- Supports square matrices
- Supports custom Block and Grid configurations for GPU
- Has switch for cuBLAS usage in PRG2

---
## Pre-requirements
- CUDA Toolkit 12.6.3
- CUDA compatible GPU (tested with CC 8.9)
- C/C++ Compiler (msvc)
- GNU Make
- Optional: Python 3.X for input data generation

---
## Compile

Change {TARGET} in Makefile first

### Release build
```
make -j3
```

### Debug build
```
make -j3 debug
```

---
## Launch

### CPU Mode
```
make launch_cpu
```
or
```
make launch_cpu_debug
```

### GPU Mode
```
make launch_gpu
```
or
```
make launch_gpu_debug
```

---
## Clean

**DELETES** all compiled files, executables and **RESULTS**
```
make clean
```
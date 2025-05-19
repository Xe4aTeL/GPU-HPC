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

Insert data in a folder next to a program. For SIZE_N 256, 512, 1024, 2048, 4096, 8192 and 16384 name files as
\<SIZE_N\>\_\<Variable\>\_\<Data type\>.txt, for other sizes: \<Variable\>\_\<Data type\>.txt

### CPU Mode
```
prg1.exe 0 SIZE_N
```
or
```
prg2.exe 0 SIZE_N
```

### GPU Mode
```
prg1.exe 1 SIZE_N
```
or
```
prg2.exe 1 SIZE_N
```

---
## Clean

**DELETES** all compiled files, executables and **RESULTS**
```
make clean
```
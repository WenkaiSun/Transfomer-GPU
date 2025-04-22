# Fast Attention

A CUDA-based implementation of Multi-Head Attention for accelerating Self-Attention inference. This project is developed as part of the CSGA-3033 course.

## Overview

This project implements an optimized version of the Multi-Head Attention mechanism using CUDA for faster inference on NVIDIA GPUs. The implementation includes custom CUDA kernels and leverages cuBLAS for efficient matrix operations.

## GPU Compatibility

The project supports various NVIDIA GPU architectures. Below is a list of supported compute capabilities and their corresponding GPUs:

| Compute Capability | GPU Models |
|-------------------|------------|
| 5.0 | GTX 750, GTX 750 Ti |
| 6.1 | GTX 1080, GTX 1070, Tesla P40 |
| 7.0 | Tesla V100 |
| 7.5 | RTX 2080 Ti, RTX 2080, RTX 2070 |
| 8.0 | NVIDIA A100 |
| 8.6 | RTX 30 series (RTX 3090, etc.) |

## Building the Project

### Prerequisites
- CUDA Toolkit
- NVIDIA GPU with compatible compute capability
- C++ compiler with C++11 support

### Compilation

Adjust the `-arch` flag according to your GPU architecture. For example, for Tesla P40 (Compute Capability 6.1):

```bash
ssh tz2693@cuda2.cims.nyu.edu
cd /home/tz2693/gpu/fast_attention
module load cuda-12.4
```

```bash
nvcc -arch=sm_61 -o mha main.cu MultiHeadAttention.cu -lcublas
```

Use this table to select the correct architecture flag:
- sm_50 for Compute Capability 5.0
- sm_61 for Compute Capability 6.1
- sm_70 for Compute Capability 7.0
- sm_75 for Compute Capability 7.5
- sm_80 for Compute Capability 8.0
- sm_86 for Compute Capability 8.6

### Running the Program

After compilation, run the executable:

```bash
./mha
```

## Performance Results

Below are the performance benchmarks showing the speedup achieved by our implementation:

![image](https://github.com/user-attachments/assets/1fdf1b91-8099-4320-b568-2ef22921b575)
![image](https://github.com/user-attachments/assets/8013a379-1728-4952-8447-40c10dc33d56)
![image](https://github.com/user-attachments/assets/b6a7ddaf-d91c-43ba-8057-8fd88b35c02a)
![image](https://github.com/user-attachments/assets/f3800a93-d24c-4cdb-a281-c782a4146832)

## Implementation Details

The project includes:
- Custom CUDA kernels for attention computation
- Optimized memory access patterns
- Efficient matrix operations using cuBLAS
- Support for half-precision (FP16) computations

## Project Structure

```
.
├── main.cu                 # Main program entry point
├── MultiHeadAttention.cu   # CUDA implementation of attention
├── MultiHeadAttention.h    # Header file with class definitions
└── error_check.h          # Error checking utilities
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Fast Attention Project Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This is a course project for CSGA-3033. Contributions and suggestions for optimization are welcome.

// error_check.h
#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define CUDA_CALL(func)                                                         \
    {                                                                           \
        cudaError_t err = (func);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in file '" << __FILE__ << "' in line "     \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "."     \
                      << std::endl;                                             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

#define CUBLAS_CALL(func)                                                       \
    {                                                                           \
        cublasStatus_t status = (func);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error in file '" << __FILE__ << "' in line "   \
                      << __LINE__ << ": " << cublasGetErrorString(status)       \
                      << "." << std::endl;                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// cuBLAS Error String
inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "<unknown>";
    }
}

#endif // ERROR_CHECK_H

// MultiHeadAttention.cu
#include "MultiHeadAttention.h"
#include "error_check.h"
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cublas_v2.h>

// Define thread block size
#define THREADS_PER_BLOCK 256

// Softmax kernel
__global__ void softmax_kernel(float *scores, int N, int num_heads, int batch_size) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || head >= num_heads || batch >= batch_size) return;

    float *scores_batch = scores + batch * num_heads * N * N;
    float *scores_head = scores_batch + head * N * N;

    // Calculate maximum value for each row
    float max_val = -1e20f;
    for (int i = 0; i < N; ++i) {
        float val = scores_head[row * N + i];
        if (val > max_val) max_val = val;
    }

    // Calculate denominator and exponents
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float val = scores_head[row * N + i];
        val = expf(val - max_val);
        scores_head[row * N + i] = val;
        sum += val;
    }

    // Normalize
    for (int i = 0; i < N; ++i) {
        scores_head[row * N + i] /= sum;
    }
}

// concat_heads kernel function
__global__ void concat_heads(float *attention_output, float *output, int N, int d_v, int num_heads, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * N * d_v * num_heads;
    if (idx >= total_elements) return;

    int batch = idx / (N * d_v * num_heads);
    int n = (idx % (N * d_v * num_heads)) / (d_v * num_heads); // sequence position
    int d = idx % (d_v * num_heads);                           // feature dimension

    output[batch * N * d_v * num_heads + n * d_v * num_heads + d] = attention_output[idx];
}

MultiHeadAttention::MultiHeadAttention(int batch_size, int N, int d_model, int num_heads)
    : batch_size(batch_size), N(N), d_model(d_model), num_heads(num_heads) {
    d_k = d_model / num_heads;
    d_v = d_model / num_heads;

    // Allocate memory
    allocateMemory();

    // Create cuBLAS handle
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
}

MultiHeadAttention::~MultiHeadAttention() {
    // Free memory
    freeMemory();

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

void MultiHeadAttention::allocateMemory() {
    size_t size_input = batch_size * N * d_model * sizeof(float);
    size_t size_weights = d_model * d_k * num_heads * sizeof(float);
    size_t size_output = batch_size * N * d_model * sizeof(float);
    size_t size_QKV = batch_size * N * d_k * num_heads * sizeof(float);
    size_t size_scores = batch_size * num_heads * N * N * sizeof(float);
    size_t size_attention = batch_size * N * d_v * num_heads * sizeof(float);

    CUDA_CALL(cudaMalloc((void**)&d_input, size_input));
    CUDA_CALL(cudaMalloc((void**)&d_WQ, size_weights));
    CUDA_CALL(cudaMalloc((void**)&d_WK, size_weights));
    CUDA_CALL(cudaMalloc((void**)&d_WV, size_weights));
    CUDA_CALL(cudaMalloc((void**)&d_WO, d_model * d_model * sizeof(float))); // Adjusted size
    CUDA_CALL(cudaMalloc((void**)&d_output, size_output));

    // Allocate memory for Q, K, V, scores, attention_output
    CUDA_CALL(cudaMalloc((void**)&d_Q, size_QKV));
    CUDA_CALL(cudaMalloc((void**)&d_K, size_QKV));
    CUDA_CALL(cudaMalloc((void**)&d_V, size_QKV));
    CUDA_CALL(cudaMalloc((void**)&d_scores, size_scores));
    CUDA_CALL(cudaMalloc((void**)&d_attention_output, size_attention));
}

void MultiHeadAttention::freeMemory() {
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_WQ));
    CUDA_CALL(cudaFree(d_WK));
    CUDA_CALL(cudaFree(d_WV));
    CUDA_CALL(cudaFree(d_WO));
    CUDA_CALL(cudaFree(d_output));

    CUDA_CALL(cudaFree(d_Q));
    CUDA_CALL(cudaFree(d_K));
    CUDA_CALL(cudaFree(d_V));
    CUDA_CALL(cudaFree(d_scores));
    CUDA_CALL(cudaFree(d_attention_output));

}

void MultiHeadAttention::initializeWeights(float *h_WQ, float *h_WK, float *h_WV, float *h_WO) {
    size_t size_weights = d_model * d_k * num_heads * sizeof(float);

    // Initialize WQ
    if (h_WQ) {
        CUDA_CALL(cudaMemcpy(d_WQ, h_WQ, size_weights, cudaMemcpyHostToDevice));
    } else {
        float *h_WQ_random = (float*)malloc(size_weights);
        for (int i = 0; i < d_model * d_k * num_heads; ++i) {
            h_WQ_random[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        CUDA_CALL(cudaMemcpy(d_WQ, h_WQ_random, size_weights, cudaMemcpyHostToDevice));
        free(h_WQ_random);
    }

    // Initialize WK
    if (h_WK) {
        CUDA_CALL(cudaMemcpy(d_WK, h_WK, size_weights, cudaMemcpyHostToDevice));
    } else {
        float *h_WK_random = (float*)malloc(size_weights);
        for (int i = 0; i < d_model * d_k * num_heads; ++i) {
            h_WK_random[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        CUDA_CALL(cudaMemcpy(d_WK, h_WK_random, size_weights, cudaMemcpyHostToDevice));
        free(h_WK_random);
    }

    // Initialize WV
    if (h_WV) {
        CUDA_CALL(cudaMemcpy(d_WV, h_WV, size_weights, cudaMemcpyHostToDevice));
    } else {
        float *h_WV_random = (float*)malloc(size_weights);
        for (int i = 0; i < d_model * d_k * num_heads; ++i) {
            h_WV_random[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        CUDA_CALL(cudaMemcpy(d_WV, h_WV_random, size_weights, cudaMemcpyHostToDevice));
        free(h_WV_random);
    }

    // Initialize WO
    size_t size_WO = d_model * d_model * sizeof(float);
    if (h_WO) {
        CUDA_CALL(cudaMemcpy(d_WO, h_WO, size_WO, cudaMemcpyHostToDevice));
    } else {
        float *h_WO_random = (float*)malloc(size_WO);
        for (int i = 0; i < d_model * d_model; ++i) {
            h_WO_random[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        CUDA_CALL(cudaMemcpy(d_WO, h_WO_random, size_WO, cudaMemcpyHostToDevice));
        free(h_WO_random);
    }

}

void MultiHeadAttention::forward() {
    // Copy input to device
    size_t size_input = batch_size * N * d_model * sizeof(float);
    CUDA_CALL(cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice));

    // Linear transformation
    linearTransform();

    // Scaled dot-product attention
    scaledDotProductAttention();

    // Concatenate multi-head outputs
    concatenateHeads();

    // Copy output back to host (if needed)
    size_t size_output = batch_size * N * d_model * sizeof(float);
    CUDA_CALL(cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost));
}

void MultiHeadAttention::linearTransform() {
    // Use cuBLAS for linear transformation, computing Q = X * WQ, K = X * WK, V = X * WV

    int batchCount = batch_size * num_heads;
    int m = N;            // number of rows in input
    int n = d_k;          // output dimension per head
    int k = d_model;      // number of columns in input

    float alpha = 1.0f;
    float beta = 0.0f;

    // Stride calculations
    long long int strideA = n * k;                      // Stride between WQ matrices (same for all heads)
    long long int strideB = N * d_model;                // Stride between input matrices (one per batch)
    long long int strideC = N * n;                      // Stride between Q matrices (per head per batch)

    // Compute Q = X * WQ
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_WQ, n, strideA,
        d_input, k, strideB,
        &beta,
        d_Q, n, strideC,
        batchCount
    );

    // Compute K = X * WK
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_WK, n, strideA,
        d_input, k, strideB,
        &beta,
        d_K, n, strideC,
        batchCount
    );

    // Compute V = X * WV
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_WV, n, strideA,
        d_input, k, strideB,
        &beta,
        d_V, n, strideC,
        batchCount
    );
}

void MultiHeadAttention::scaledDotProductAttention() {
    // Compute attention scores: scores = Q * K^T / sqrt(d_k)
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    float alpha = scale;
    float beta = 0.0f;

    int batchCount = batch_size * num_heads;
    int m = N;    // rows in Q
    int n = N;    // columns in K^T
    int k = d_k;  // shared dimension

    // Stride calculations
    long long int strideA = N * k;
    long long int strideB = N * k;
    long long int strideC = N * N;

    // Compute scores = Q * K^T
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,  // K^T
        CUBLAS_OP_N,  // Q
        n, m, k,
        &alpha,
        d_K, k, strideB,
        d_Q, k, strideA,
        &beta,
        d_scores, n, strideC,
        batchCount
    );

    // Apply softmax to scores
    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;
    dim3 grid(blocks, num_heads, batch_size);
    softmax_kernel<<<grid, threads>>>(d_scores, N, num_heads, batch_size);
    CUDA_CALL(cudaGetLastError());

    // Compute attention_output = scores * V
    m = N;        // rows in scores
    n = d_v;      // columns in V
    k = N;        // columns in scores

    alpha = 1.0f;
    beta = 0.0f;

    strideA = N * N;
    strideB = N * d_v;
    strideC = N * d_v;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_V, n, strideB,
        d_scores, k, strideA,
        &beta,
        d_attention_output, n, strideC,
        batchCount
    );
}

void MultiHeadAttention::concatenateHeads() {
    // Concatenate the outputs from multiple heads into a matrix of size [batch_size, N, d_model]
    int total_elements = batch_size * N * d_v * num_heads;
    int threads = THREADS_PER_BLOCK;
    int blocks = (total_elements + threads - 1) / threads;
    concat_heads<<<blocks, threads>>>(d_attention_output, d_output, N, d_v, num_heads, batch_size);
    CUDA_CALL(cudaGetLastError());

    // Linear transformation with WO matrix
    int m = N * batch_size;
    int n = d_model;
    int k = d_model;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_WO, n,
        d_output, k,
        &beta,
        d_output, n
    );
}

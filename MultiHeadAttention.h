// MultiHeadAttention.h

#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

class MultiHeadAttention {
public:
    MultiHeadAttention(int N, int d_model, int num_heads);
    ~MultiHeadAttention();

    void initializeWeights(float *h_WQ = nullptr, float *h_WK = nullptr, float *h_WV = nullptr, float *h_WO = nullptr);
    void forward();

    // Host pointers
    float *h_input;

    // Parameters
    int N;          // Sequence length
    int d_model;    // Model dimension
    int num_heads;  // Number of heads
    int batch_size; // Batch size
    int d_k;        // Dimension of keys and queries per head
    int d_v;        // Dimension of values per head

    // Device pointers (made public for testing purposes)
    float *d_input;
    float *d_WQ;
    float *d_WK;
    float *d_WV;
    float *d_WO;
    float *d_output;

    // Additional Device pointers
    float *d_Q;
    float *d_K;
    float *d_V;
    float *d_scores;
    float *d_attention_output;
    float *d_gamma;
    float *d_beta;

    // Methods (made public for testing purposes)
    void allocateMemory();
    void freeMemory();
    void linearTransform();
    void scaledDotProductAttention();
    void concatenateHeads();
    void addResidualConnection();
    void layerNormalization();

private:
    cublasHandle_t handle;
};

#endif // MULTI_HEAD_ATTENTION_H

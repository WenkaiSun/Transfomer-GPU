// main.cu
#include "MultiHeadAttention.h"
#include "error_check.h"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Define model parameters
    const int N = 128;         // Sequence length
    const int d_model = 512;   // Model dimension
    const int num_heads = 8;   // Number of attention heads
    
    // Create MultiHeadAttention instance
    MultiHeadAttention mha(N, d_model, num_heads);
    
    // Initialize input with random values
    mha.h_input = (float*)malloc(N * d_model * sizeof(float));
    for (int i = 0; i < N * d_model; ++i) {
        mha.h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Initialize model weights
    mha.initializeWeights();
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CALL(cudaEventRecord(start, 0));
    
    // Forward pass through the attention layer
    mha.forward();
    
    // Record end event
    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    
    // Calculate and print execution time
    float elapsedTime;
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Execution time: " << elapsedTime << " ms" << std::endl;
    
    // Cleanup CUDA events
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    
    // Optional: Print output values
    // Commented out to avoid cluttering the console
    /*
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d_model; ++j) {
            std::cout << mha.h_input[i * d_model + j] << " ";
        }
        std::cout << std::endl;
    }
    */ 
    
    // Free host memory
    free(mha.h_input);
    return 0;
}

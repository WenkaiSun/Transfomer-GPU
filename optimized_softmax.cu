#include <cuda.h>

__global__ void optimized_softmax(float* A, float* B, int N) {
  int tx = threadIdx.x; // 1
  int ty = threadIdx.y; // block size
  int bx = blockIdx.x; // N
  int by = blockIdx.y; // N / block size

  int r = blockDim.x * bx + tx;
  int c = blockDim.y * by + ty;

  float thread = 0;
  __shared__ float thread_sum[TILE_SIZE];

  for (int i = 0; i < N; i += TILE_SIZE) {
    if ((i + ty) < N) {
      thread += expf(A[r*N + i + ty]);
    }
  }
  thread_sum[ty] = thread;
  __syncthreads();

  float sumexp = 0;
  for (int j = 0; j < TILE_SIZE; j++) {
    sumexp += thread_sum[j];
  }

  B[r*N + c] = expf(A[r*N + c])/sumexp;
}
#include <cuda.h>

//https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
__global__ void transpose(float* A, float* B, int width, int height)
{
  int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float block[TILE_DIM][TILE_DIM+1];
	
  block[threadIdx.y][threadIdx.x] = A[yIndex * width + xIndex;];
	__syncthreads();

	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;
  B[yIndex * height + xIndex] = block[threadIdx.x][threadIdx.y];
}
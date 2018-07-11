#include<stdio.h>

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

// Kernel
__global__ void print(int n) {

	// Index
	int tIdx = threadIdx.x;
	int bIdx = blockIdx.x;

	// Dimension
	int bDim = blockDim.x; 
	int gDim = gridDim.x;

	printf("Grid-Dimension: %i, Block-Dimension: %i\n Block-ID: %i, Thread-ID: %i\n\n", gDim, bDim, bIdx, tIdx);


	//for (int i = 0; i < n; i++) {
	//	printf("Grid: %i, Block: %i, Thread: %i\n", i,i,i);
	//}
}


// Main
int main(void) {
	int n = 25;

	print<<<2048, 1>>>(n);
	cudaCheckError();

	// Synx Device
	cudaDeviceSynchronize();

	return 0;
}

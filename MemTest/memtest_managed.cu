/*
  Test Programm nach diesmal mit Managed Memory:
  https://www.thomas-krenn.com/de/wiki/CUDA_Programmierung
*/
#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

// Vars
// Device/Host-Vars
int* A;
int* B;
int* C;

// Prototypes
void RandomInit(int* data, int n);
int CheckResults(int* A, int* B, int* C, int n);

// Kernel
__global__ void VecAdd(const int* A, const int* B, int* C, int N) {

	// Index holen
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N)
		C[i] = A[i] + B[i];

}

int main(void) {
	
	printf("Vector addtion\n");
	//int i;
	int N = 100000 * 1000;
	size_t size = N * sizeof(int);

	// Speicher auf Device allozieren
	cudaMallocManaged(&A, size);
	cudaMallocManaged(&B, size);
	cudaMallocManaged(&C, size);

	// Random Init
	RandomInit(A, N);
	RandomInit(B, N);

	// Kernelaufruf
	// Nvidia GTX 1080 TI hat 1024 Threads pro Block
	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	printf("BlocksPerGrid = %i, ThreadsPerBlock = %i\n\n", blocksPerGrid, threadsPerBlock);

	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

	// Auf das Gerät warten
	cudaDeviceSynchronize();

	// Ergebnisse prüfen
	if (CheckResults(A, B, C, N) == 0)
		printf("Alles ok!\n");
	else
		printf("Fehler\n");

	// Speicherfreigabe
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	
	return 0;
}

// Vector mit Zufallszahlen füllen
void RandomInit(int* data, int n) {
	for (int i = 0; i < n; i++) 
		data[i] = rand() % (int) 100;
}

// Ergebnis Prüfen
int CheckResults(int* A, int* B, int* C, int n) {
	int i;
	for (i = 0; i < n; i++) {
		if ((A[i]+B[i]) != C[i])
			return -1;
	}
	return 0;
}

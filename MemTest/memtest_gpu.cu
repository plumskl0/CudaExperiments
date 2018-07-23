/*
  Test Programm nach:
  https://www.thomas-krenn.com/de/wiki/CUDA_Programmierung
*/
#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

// Vars
// Host-Vars
int* h_A;
int* h_B;
int* h_C;

// Device-Vars
int* d_A;
int* d_B;
int* d_C;

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

	// Speicher auf Host allozieren
	h_A = (int*)malloc(size);
	h_B = (int*)malloc(size);
	h_C = (int*)malloc(size);

	// Random Init
	RandomInit(h_A, N);
	RandomInit(h_B, N);

	// Speicher auf Device allozieren
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Vektoren zum Device kopieren
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Kernelaufruf
	// Nvidia GTX 1080 TI hat 1024 Threads pro Block
	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	printf("BlocksPerGrid = %i, ThreadsPerBlock = %i\n\n", blocksPerGrid, threadsPerBlock);

	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

	// Auf das Gerät warten
	cudaDeviceSynchronize();

	// Ergebnis auf Host kopieren
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Ergebnisse prüfen
	if (CheckResults(h_A, h_B, h_C, N) == 0)
		printf("Alles ok!\n");
	else
		printf("Fehler\n");

	// Speicherfreigabe
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	free(h_A);
	free(h_B);
	free(h_C);

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

/*
  Test Programm nach aber nur auf der CPU:
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

// Prototypes
void RandomInit(int* data, int n);
int CheckResults(int* A, int* B, int* C, int n);

// Kernel
void VecAdd(const int* A, const int* B, int* C, int N) {

	for (int i = 0; i < N; i++)
		C[i] = A[i] + B[i];

}

int main(void) {
	
	printf("Vector addtion\n");
	//int i;
	int N = 100000 * 10000;
	size_t size = N * sizeof(int);

	// Speicher auf Host allozieren
	h_A = (int*)malloc(size);
	h_B = (int*)malloc(size);
	h_C = (int*)malloc(size);

	// Random Init
	RandomInit(h_A, N);
	RandomInit(h_B, N);

	// Kernelaufruf
	VecAdd(h_A, h_B, h_C, N);

	// Ergebnisse prüfen
	if (CheckResults(h_A, h_B, h_C, N) == 0)
		printf("Alles ok!\n");
	else
		printf("Fehler\n");

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

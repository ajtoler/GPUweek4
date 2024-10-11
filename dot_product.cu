#include <stdio.h>
#include <time.h>


#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {

	// Use atomicAdd	
    int idx =  threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) atomicAdd(C, A[idx] * B[idx]);
}


int main() {
	
	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
	}


	*h_C = 0;


	// Allocate device memory 
	cudaMalloc(&d_A, DSIZE*sizeof(int));
	cudaMalloc(&d_B, DSIZE*sizeof(int));
	cudaMalloc(&d_C, DSIZE*sizeof(int));

	// Check memory allocation for errors
	cudaCheckErrors();

	// Copy the matrices on GPU
	cudaMemcpy(d_A, h_A, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, DSIZE*sizeof(int), cudaMemcpyHostToDevice);

	// Check memory copy for errors
    cudaCheckErrors();

	// Define block/grid dimensions and launch kernel
	int block = BLOCK_SIZE;
	int grid((DSIZE + block - 1) / block);
	dot_product<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

	// Copy results back to host
	cudaMemcpy(h_C, d_C, DSIZE*sizeof(int), cudaMemcpyDeviceToHost);

    // Check copy for errors
    cudaCheckErrors();

	// Verify result
    // for (int i = 0; i < DSIZE; i++) printf("A[%d]: %d, B[%d]: %d\n", i, h_A[i], i, h_B[i]);
	printf("Dot Product C: %d\n", *h_C);

	// Free allocated memory
	free(h_A);
    free(h_B);
	free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
	cudaFree(d_C);
	
	return 0;

}
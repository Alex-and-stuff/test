#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>

#define SIZE 256
#define SHMEM_SIZE 256 

__global__ void sum_reduction(int* v, int* v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
		printf("result: %d\n", partial_sum[0]);
	}
}

void initialize_vector(int* v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 2;//rand() % 10;
	}
}

__global__ void initCurand(curandState* state, unsigned long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, idx, 0, &state[idx]);
}
__global__ void uniformRand(curandState* state, float* rand) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	rand[idx] = curand_uniform(&state[idx])-0.5;  // Generate the random number from 0.0-1.0
	printf("%.1f\n", rand[idx]);
}
#define DSIZE 10000
int main() {
	// Setup parameters and Initialize variables
	float* dev_X0;
	float* dev_U;
	float* dev_E;
	float* dev_q;
	float* dev_qmin;

	float host_X0[] = { 0.0f, 0.0f, 0.0f, 2.0f };
	float host_U0[] = { 0.0f, 0.0f };

	float* h_a1, * d_a1;
	curandState* devState;

	h_a1 = (float*)malloc(DSIZE * sizeof(float));
	cudaMalloc((void**)&d_a1, DSIZE * sizeof(float));
	cudaMalloc((void**)&devState, DSIZE * sizeof(curandState));

	initCurand << <100, 100 >> > (devState, 1);
	cudaDeviceSynchronize();
	uniformRand << <100, 100 >> > (devState, d_a1);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a1, d_a1, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

	printf("returned random value is %.1f %.1f %.1f\n", h_a1[0],h_a1[1],h_a1[2]);



	//// Vector size
	//int n = 1 << 16;
	//size_t bytes = n * sizeof(int);

	//// Original vector and result vector
	//int* h_v, * h_v_r;
	//int* d_v, * d_v_r;

	//// Allocate memory
	//h_v = (int*)malloc(bytes);
	//h_v_r = (int*)malloc(bytes);
	//cudaMalloc(&d_v, bytes);
	//cudaMalloc(&d_v_r, bytes);

	//// Initialize vector
	//initialize_vector(h_v, n);

	//// Copy to device
	//cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	//// TB Size
	//int TB_SIZE = SIZE;

	//// Grid Size (cut in half) (No padding)
	//int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE / 2;

	//// Call kernel
	//sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	//sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	//// Copy to host;
	//cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	//// Print the result
	////printf("Accumulated result is %d \n", h_v_r[0]);
	////scanf("Press enter to continue: ");
	//printf("result: %d\n", h_v_r[0]);
	//assert(h_v_r[0] == 65536 * 2);

	//printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}

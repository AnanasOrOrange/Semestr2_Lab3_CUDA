#include "cuda_runtime.h"					
#include "device_launch_parameters.h"		
#include <iostream>
#include <chrono>

#define CHECK_ERR()														
void check() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\M", __FILE__, __LINE__, cudaGetErrorString(err));
        system("pause");
        exit(1);
    }
}

// число разбиений
#define N 100
// время, до которого считать
#define T 0.05

#define BLOCK_SIZE 16

// для вывода матриц
//#define OUT

// упрощение для удобства
#define node(i,j) ((i) * (N) + (j))
#define Q(i,j) double((i) == (N) / 2 && (j) == (N) / 2)
#define isBorder(i,j) bool((i) == (0) || (j) == (0) || (i) == (N) - 1 || (j) == (N) - 1)

#define hx (1.0 / double((N) - 1))
#define hy (1.0 / double((N) - 1))

#define tau (hx * hx * hy * hy / (3.0 * (hx * hx + hy * hy)))

void initMesh(double* mesh) {
    for (int i = 0; i < N * N; i++) {
        mesh[i] = 0.0;
    }
}

void CPU_Cross(double*& prev, double*& next) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!isBorder(i, j)) {
                next[node(i, j)] = prev[node(i, j)] + tau * ( (prev[node(i + 1, j)] - 2.0 * prev[node(i, j)] + prev[node(i - 1, j)]) / hx / hx +   \
                                                              (prev[node(i, j + 1)] - 2.0 * prev[node(i, j)] + prev[node(i, j - 1)]) / hy / hy + Q(i, j));                                                             
            }
        }
    }
}

__global__
void GPU_Cross(double* prev, double* next) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        int i = tid / N;
        int j = tid % N;

        if (tid < N * N && !isBorder(i, j)) {
            next[node(i, j)] = prev[node(i, j)] + tau * ((prev[node(i + 1, j)] - 2.0 * prev[node(i, j)] + prev[node(i - 1, j)]) / hx / hx + \
                                                         (prev[node(i, j + 1)] - 2.0 * prev[node(i, j)] + prev[node(i, j - 1)]) / hy / hy + Q(i, j));
        }
}

#ifdef OUT
void printMesh(double* mesh) {
    std::cout << std::endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mesh[node(i, j)] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}
#endif

bool isEquel(double* mesh1, double* mesh2) {
    for (int i = 0; i < N * N; i++) {
        if (mesh1[i] != mesh2[i]) {
            return false;
        }
    }
    return true;
}


int main()
{
    // CPU
    double* prevCPU = new double[N * N];
    double* nextCPU = new double[N * N];
    initMesh(prevCPU);
    initMesh(nextCPU);

    std::cout << "N\t" << N << std::endl;
    std::cout << "T\t" << T << std::endl;
    std::cout << std::endl;
    std::cout << "tau\t" << tau << std::endl;
    std::cout << "hx\t" << hx << std::endl;
    std::cout << "hy\t" << hy << std::endl;
    std::cout << std::endl;

    auto startCPU = std::chrono::steady_clock::now();

    for (double t = 0.0; t < T; t += tau) {
        CPU_Cross(prevCPU, nextCPU);
        std::swap(prevCPU, nextCPU);
    }

    auto endCPU = std::chrono::steady_clock::now();
    std::cout << "CPU time\t= " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << "\t\tmillisec, (1e-3 sec)" << std::endl;

#ifdef OUT
    printMesh(prevCPU);
#endif

    // CUDA
    double* prevGPU = new double[N * N];
    double* nextGPU = new double[N * N];
    initMesh(prevGPU);
    initMesh(nextGPU);

    double* prevDev, * nextDev;

    cudaMalloc((void**)&prevDev, N * N * sizeof(double)); CHECK_ERR();
    cudaMalloc((void**)&nextDev, N * N * sizeof(double)); CHECK_ERR();

    cudaEvent_t startGPU, stopGPU;

    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU);

	cudaMemcpy(prevDev, prevGPU, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(nextDev, nextGPU, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR();

    for (double t = 0.0; t < T; t += tau) {
        GPU_Cross <<< N * N / BLOCK_SIZE + 1, BLOCK_SIZE>>>(prevDev, nextDev); CHECK_ERR();
        std::swap(prevDev, nextDev);
    }

    cudaMemcpy(prevGPU, prevDev, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR();
    //cudaMemcpy(nextGPU, nextDev, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float timeCUDA = 0;
    cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

    std::cout << "GPU time\t= " << timeCUDA << "\tmillisec, (1e-3 sec)" << std::endl;

#ifdef OUT
    printMesh(prevGPU);
#endif
    std::cout << std::endl;
    std::cout << "Checking results:" << std::endl;
    if (!isEquel(prevCPU, prevGPU)) {
		std::cout << "ERROR" << std::endl;
	}
    else {
		std::cout << "OK" << std::endl;
    }
}


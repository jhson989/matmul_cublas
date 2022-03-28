#include "../include/kernel.cuh"

void run_cuBLAS(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, const int M, const int N, const int K, const float alpha, const float beta) {

    printf("[Kernel] Run kernal\n");

    /*** Initialize device memory ***/
    float *d_A, *d_B, *d_C;
    cudaErrChk( cudaMalloc((void**)(&d_A), sizeof(float)*M*K) );
    cudaErrChk( cudaMalloc((void**)(&d_B), sizeof(float)*K*N) );
    cudaErrChk( cudaMalloc((void**)(&d_C), sizeof(float)*M*N) );
    cudaErrChk( cudaMemcpy(d_A, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_B, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() )


    /*** Setup cuBLAS execution handler ***/
    cublasHandle_t handle;
    cublasErrChk (cublasCreate (&handle));


    /*** Run CUDA kernel ***/
    
    // Record events for performance measurement
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );

    // Run cuBLAS kernel
    cublasErrChk( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N) );

    // End of events
    cudaErrChk(cudaEventRecord(stop, NULL));
    cudaErrChk(cudaEventSynchronize(stop));
    float msec_total = 0.0f;
    float gflo = 2.0f*M*N*K*1e-9; // multiply and add
    cudaErrChk(cudaEventElapsedTime(&msec_total, start, stop));
    printf(" -- elaped time: %.4f sec\n", msec_total*1e-3);
    printf(" -- gFlops : %.4f gflops\n", gflo/(msec_total*1e-3));

    cudaErrChk( cudaMemcpy(C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() )


    /*** Finalize ***/
    cudaErrChk( cudaFree(d_A) );
    cudaErrChk( cudaFree(d_B) );
    cudaErrChk( cudaFree(d_C) );
    cublasErrChk( cublasDestroy(handle) );
    
}

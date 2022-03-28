#include "../include/debug.cuh"

const char* cuBLAS_get_error_enum(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


void cuBLAS_assert(cublasStatus_t code, const char *file, int line) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"cuBLAS assert: %s %s %d\n", cuBLAS_get_error_enum(code), file, line);
      if (abort) exit(code);
   }
}

void cuda_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void check_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int M, const int N, const int K){

    printf("[TEST] Test start..\n");

    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            float sum = 0.0f;
            for (int k=0; k<K; k++) {
                sum += (A[y*K+k]*B[k*N+x]);
            }

            // Error tolerance
            if (C[y*N+x] >= sum+1e-5 || C[y*N+x] <= sum-1e-5) {
                printf(" -- [ERROR] C[%d,%d] = %f != gt(%f)\n", y, x, C[y*N+x], sum);
                printf(" -- test failed...!\n");
                return;
            }
        }
    }

    printf(" -- test passed !!\n");
    return;
}

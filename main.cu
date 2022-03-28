
#include <random>
#include <sys/time.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include "include/debug.cuh"
#include "include/kernel.cuh"


/*********************************************************
  * Configuration
  ********************************************************/
#define DEBUG_ON
const int M = 1024;
const int N = 1024;
const int K = 1024;
const float alpha = 1.0f;
const float beta = 0.0f;

/*********************************************************
  * Helper functions
  ********************************************************/
inline float get_random_number() {return 1.0f*(std::rand()%11-5)/2.0f;}



/*********************************************************
  * cuBLAS interface functions
  ********************************************************/



/***************************************
  * Main function
  **************************************/
int main(int argc, char** argv) {

    /*** Program configuration ***/
    srand(0);
    printf("\n================================================\n");
    printf("cuBLAS GEMM Example for FP32 MatMul\n");
    printf(" -- GEMM : C[a, c] = alpha * A[a, b] @ B[b, c] + beta * C[a, c]\n");
    printf(" -- C[%d, %d] = %f * A[%d, %d] @ B[%d, %d] + %f * C[%d, %d]\n", M,N,1.0f,M,K,K,N,0.0f,M,N);
    printf(" -- total size of matrices : %.3f GB\n", 1.0f*(M*N+M*K+K*N)*sizeof(float)*1e-9);
    printf("================================================\n\n");

    /*** Initialize Data ***/
    std::vector<float> A(M*K);
    std::generate(A.begin(), A.end(), get_random_number);
    std::vector<float> B(K*N);
    std::generate(B.begin(), B.end(), get_random_number);
    std::vector<float> C(M*N, 0);

    /*** Run matmul ***/
    run_cuBLAS(A, B, C, M, N, K, alpha, beta);

    /*** Test result ***/
    #ifdef DEBUG_ON
    check_result(A, B, C, M, N, K);
    #endif


    /*** Finalize ***/

    return 0;
}


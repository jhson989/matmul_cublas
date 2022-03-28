#pragma once

#include <vector>
#include <iostream>
#include <cublas_v2.h>

// Output error messages
const char* cuBLAS_get_error_enum(cublasStatus_t error);
void cuBLAS_assert(cublasStatus_t code, const char *file, int line);
void cuda_assert(cudaError_t code, const char *file, int line);

#define cublasErrChk(ans) { cuBLAS_assert((ans), __FILE__, __LINE__); }
#define cudaErrChk(ans) { cuda_assert((ans), __FILE__, __LINE__); }


// Checking result
void check_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int M, const int N, const int K);


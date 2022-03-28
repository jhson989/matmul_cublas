#pragma once

#include "debug.cuh"

void run_cuBLAS(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, const int M, const int N, const int K, const float alpha, const float beta);

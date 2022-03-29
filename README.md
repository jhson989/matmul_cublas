# cuBLAS GEMM Example for FP32 MatMul

## 1. Introduction
설명을 https://computing-jhson.tistory.com/24 에 추가함
- main.cu : main 함수
- src/kernel.cu : cuBLAS api 호출 interface 함수
- src/debug.cu : CUDA 및 cuBLAS 에러 핸들링 및 계산 결과 검증을 위한 debugging functions

## 2. How to Run
- make
    - make DEBUG=ON 시, cuBLAS 계산 결과 검증 진행
- make run
    - main.cu 내 M, N, K 변수를 통해 Matrix 크기를 조절할 수 있음

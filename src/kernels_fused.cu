#include "kernels_variants.cuh"

namespace fused {

// Reuse shared implementations for non-fused parts
void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    shared::matrixMultiply(A, B, C);
}
void addBias(Matrix& Y, const Matrix& b) {
    shared::addBias(Y, b);
}
void reluActivation(const Matrix& Z, Matrix& A) {
    shared::reluActivation(Z, A);
}
void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C) {
    shared::matrixMultiplyTransposeA(A, B, C);
}
void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C) {
    shared::matrixMultiplyTransposeB(A, B, C);
}
void computeBiasGradient(const Matrix& dY, Matrix& db) {
    shared::computeBiasGradient(dY, db);
}
void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX) {
    shared::reluBackward(dY, Z, dX);
}
void updateWeights(Matrix& W, const Matrix& dW, float lr) {
    shared::updateWeights(W, dW, lr);
}
void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss) {
    shared::computeMSEGradient(P, Y, d_loss);
}
void softmaxActivation(const Matrix& Z, Matrix& A) {
    shared::softmaxActivation(Z, A);
}

#define TILE_SIZE 16

__global__ void matmul_with_bias_relu_kernel(const float* A, const float* B, const float* b, float* C, int m, int k, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + tx;
        if (row < m && A_col < k) As[ty][tx] = A[row * k + A_col];
        else As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        if (B_row < k && col < n) Bs[ty][tx] = B[B_row * n + col];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        float val = sum + b[col];
        C[row * n + col] = fmaxf(0.0f, val);
    }
}

void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_with_bias_relu_kernel<<<gridDim, blockDim>>>(A.data, B.data, b.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_with_bias_kernel(const float* A, const float* B, const float* b, float* C, int m, int k, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + tx;
        if (row < m && A_col < k) As[ty][tx] = A[row * k + A_col];
        else As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        if (B_row < k && col < n) Bs[ty][tx] = B[B_row * n + col];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum + b[col];
    }
}

void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_with_bias_kernel<<<gridDim, blockDim>>>(A.data, B.data, b.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace fused

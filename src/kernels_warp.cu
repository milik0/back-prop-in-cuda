#include "kernels_variants.cuh"
#include <cfloat>

namespace warp_opt {

// Reuse fused implementations
void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    fused::matrixMultiplyWithBiasAndReLU(A, B, b, C);
}
void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    fused::matrixMultiplyWithBias(A, B, b, C);
}
void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    fused::matrixMultiply(A, B, C);
}
void addBias(Matrix& Y, const Matrix& b) {
    fused::addBias(Y, b);
}
void reluActivation(const Matrix& Z, Matrix& A) {
    fused::reluActivation(Z, A);
}
void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C) {
    fused::matrixMultiplyTransposeA(A, B, C);
}
void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C) {
    fused::matrixMultiplyTransposeB(A, B, C);
}
void computeBiasGradient(const Matrix& dY, Matrix& db) {
    fused::computeBiasGradient(dY, db);
}
void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX) {
    fused::reluBackward(dY, Z, dX);
}
void updateWeights(Matrix& W, const Matrix& dW, float lr) {
    fused::updateWeights(W, dW, lr);
}
void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss) {
    fused::computeMSEGradient(P, Y, d_loss);
}

// Warp Reduce Sum
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp Reduce Max
__device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void softmax_warp_kernel(const float* Z, float* A, int m, int n) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    // Each warp handles one row.
    // threadIdx.x is the lane index.
    
    if (row < m) {
        float max_val = -FLT_MAX;
        
        // 1. Find Max
        for (int i = threadIdx.x; i < n; i += warpSize) {
            max_val = fmaxf(max_val, Z[row * n + i]);
        }
        max_val = warpReduceMax(max_val);
        // Broadcast max_val to all lanes
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        // 2. Compute Exp and Sum
        float sum_exp = 0.0f;
        for (int i = threadIdx.x; i < n; i += warpSize) {
            sum_exp += expf(Z[row * n + i] - max_val);
        }
        sum_exp = warpReduceSum(sum_exp);
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

        // 3. Normalize and Write
        for (int i = threadIdx.x; i < n; i += warpSize) {
            A[row * n + i] = expf(Z[row * n + i] - max_val) / sum_exp;
        }
    }
}

void softmaxActivation(const Matrix& Z, Matrix& A) {
    // One warp per row
    dim3 blockDim(32, 4); // 4 warps per block, handling 4 rows
    dim3 gridDim((Z.rows + blockDim.y - 1) / blockDim.y);
    
    softmax_warp_kernel<<<gridDim, blockDim>>>(Z.data, A.data, Z.rows, Z.cols);
    CHECK_CUDA(cudaGetLastError());
}

}

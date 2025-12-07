#include "kernels.cuh"
#include <cmath>
#include <cfloat>

// CUDA Kernel for Matrix Multiplication
// C = A * B
// A: m x k, B: k x n, C: m x n
__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // A: m x k, B: k x n
    // C must be m x n
    if (A.cols != B.rows) {
        std::cerr << "Error: Matrix dimensions mismatch for multiplication." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        std::cerr << "Error: Output matrix dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);

    matmul_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for Adding Bias
// Y: m x n, b: 1 x n
// Each row of Y corresponds to a sample, we add b to each sample.
__global__ void add_bias_kernel(float* Y, const float* b, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        Y[row * n + col] += b[col];
    }
}

void addBias(Matrix& Y, const Matrix& b) {
    if (Y.cols != b.cols) {
        std::cerr << "Error: Bias dimension mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    // Assuming b is a vector (1 row or 1 col, but treated as size n)
    
    dim3 blockDim(16, 16);
    dim3 gridDim((Y.cols + blockDim.x - 1) / blockDim.x, (Y.rows + blockDim.y - 1) / blockDim.y);

    add_bias_kernel<<<gridDim, blockDim>>>(Y.data, b.data, Y.rows, Y.cols);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for ReLU
__global__ void relu_kernel(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

void reluActivation(const Matrix& Z, Matrix& A) {
    if (Z.rows != A.rows || Z.cols != A.cols) {
        std::cerr << "Error: Dimension mismatch for ReLU." << std::endl;
        exit(EXIT_FAILURE);
    }
    int size = Z.rows * Z.cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(Z.data, A.data, size);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for Softmax (Naive implementation)
// This is not optimized. Optimized version would use shared memory reductions.
__global__ void softmax_kernel(const float* Z, float* A, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        // 1. Find max for stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < n; ++i) {
            float val = Z[row * n + i];
            if (val > max_val) max_val = val;
        }

        // 2. Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum_exp += expf(Z[row * n + i] - max_val);
        }

        // 3. Normalize
        for (int i = 0; i < n; ++i) {
            A[row * n + i] = expf(Z[row * n + i] - max_val) / sum_exp;
        }
    }
}

void softmaxActivation(const Matrix& Z, Matrix& A) {
    if (Z.rows != A.rows || Z.cols != A.cols) {
        std::cerr << "Error: Dimension mismatch for Softmax." << std::endl;
        exit(EXIT_FAILURE);
    }
    // One thread per row
    int threadsPerBlock = 256;
    int blocksPerGrid = (Z.rows + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(Z.data, A.data, Z.rows, Z.cols);
    CHECK_CUDA(cudaGetLastError());
}

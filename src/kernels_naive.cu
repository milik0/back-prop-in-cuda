#include "kernels_variants.cuh"
#include <cfloat>

namespace naive {

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
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void add_bias_kernel(float* Y, const float* b, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        Y[row * n + col] += b[col];
    }
}

void addBias(Matrix& Y, const Matrix& b) {
    dim3 blockDim(16, 16);
    dim3 gridDim((Y.cols + blockDim.x - 1) / blockDim.x, (Y.rows + blockDim.y - 1) / blockDim.y);
    add_bias_kernel<<<gridDim, blockDim>>>(Y.data, b.data, Y.rows, Y.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_kernel(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

void reluActivation(const Matrix& Z, Matrix& A) {
    int size = Z.rows * Z.cols;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(Z.data, A.data, size);
    CHECK_CUDA(cudaGetLastError());
}

void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    matrixMultiply(A, B, C);
    addBias(C, b);
}

void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C) {
    matrixMultiply(A, B, C);
    addBias(C, b);
    reluActivation(C, C);
}

__global__ void matmul_transposeA_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < k && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * k + row] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_transposeA_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_transposeB_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = sum;
    }
}

void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_transposeB_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.rows);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void bias_grad_kernel(const float* dY, float* db, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        atomicAdd(&db[col], dY[row * n + col]);
    }
}

void computeBiasGradient(const Matrix& dY, Matrix& db) {
    db.zeros();
    dim3 blockDim(16, 16);
    dim3 gridDim((dY.cols + blockDim.x - 1) / blockDim.x, (dY.rows + blockDim.y - 1) / blockDim.y);
    bias_grad_kernel<<<gridDim, blockDim>>>(dY.data, db.data, dY.rows, dY.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void relu_backward_kernel(const float* dY, const float* Z, float* dX, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dX[idx] = (Z[idx] > 0.0f) ? dY[idx] : 0.0f;
    }
}

void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX) {
    int size = dY.rows * dY.cols;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(dY.data, Z.data, dX.data, size);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void update_weights_kernel(float* W, const float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}

void updateWeights(Matrix& W, const Matrix& dW, float lr) {
    int size = W.rows * W.cols;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    update_weights_kernel<<<blocks, threads>>>(W.data, dW.data, lr, size);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void mse_gradient_kernel(const float* P, const float* Y, float* d_loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_loss[idx] = P[idx] - Y[idx];
    }
}

void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss) {
    int size = P.rows * P.cols;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mse_gradient_kernel<<<blocks, threads>>>(P.data, Y.data, d_loss.data, size);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void softmax_kernel(const float* Z, float* A, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < n; ++i) {
            if (Z[row * n + i] > max_val) max_val = Z[row * n + i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i) {
            float val = expf(Z[row * n + i] - max_val);
            A[row * n + i] = val;
            sum_exp += val;
        }

        for (int i = 0; i < n; ++i) {
            A[row * n + i] /= sum_exp;
        }
    }
}

void softmaxActivation(const Matrix& Z, Matrix& A) {
    int threads = 256;
    int blocks = (Z.rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(Z.data, A.data, Z.rows, Z.cols);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace naive

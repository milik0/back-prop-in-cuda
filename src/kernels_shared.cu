#include "kernels_variants.cuh"
#include <cfloat>

#define TILE_SIZE 16

namespace shared {

__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + tx;
        int B_row = t * TILE_SIZE + ty;

        if (row < m && A_col < k) As[ty][tx] = A[row * k + A_col];
        else As[ty][tx] = 0.0f;

        if (B_row < k && col < n) Bs[ty][tx] = B[B_row * n + col];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

// Add Bias (Shared memory optimization? Not really needed for elementwise, but let's use the one from original kernels.cu)
__global__ void add_bias_kernel(float* Y, const float* b, int m, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float b_shared[TILE_SIZE];

    if (ty == 0) {
        if (col < n) b_shared[tx] = b[col];
        else b_shared[tx] = 0.0f;
    }
    __syncthreads();

    if (row < m && col < n) {
        Y[row * n + col] += b_shared[tx];
    }
}

void addBias(Matrix& Y, const Matrix& b) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((Y.cols + blockDim.x - 1) / blockDim.x, (Y.rows + blockDim.y - 1) / blockDim.y);
    add_bias_kernel<<<gridDim, blockDim>>>(Y.data, b.data, Y.rows, Y.cols);
    CHECK_CUDA(cudaGetLastError());
}

// ReLU (No shared mem benefit really, use naive)
void reluActivation(const Matrix& Z, Matrix& A) {
    naive::reluActivation(Z, A);
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
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // C is k x n
    int row = by * TILE_SIZE + ty; // Index in k (rows of C, cols of A)
    int col = bx * TILE_SIZE + tx; // Index in n (cols of C, cols of B)

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    
    // Loop over inner dimension m
    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // Load A^T tile (which is A tile transposed conceptually, or just load A properly)
        // We want A^T[row][k_idx] -> A[k_idx][row]
        // Here 'row' is the row in C, which is the col in A.
        // 'k_idx' (inner loop) corresponds to row in A.
        
        // Loading A tile:
        // We need A[t*TILE + ...][row]
        // Let's map threads to load A.
        // A is m x k.
        // We want to load a tile of A that covers rows t*TILE to (t+1)*TILE
        // and cols corresponding to 'row' (which is by*TILE + ty).
        // But 'row' varies with ty.
        
        // Standard approach for C = A^T * B:
        // Tile in A is (t*TILE, by*TILE) of size TILE x TILE.
        // We load A[t*TILE + ty][by*TILE + tx] into As[ty][tx]?
        // Let's follow kernels.cu logic exactly.
        
        int A_row = t * TILE_SIZE + ty;
        int A_col = by * TILE_SIZE + tx;
        
        if (A_row < m && A_col < k)
            As[ty][tx] = A[A_row * k + A_col];
        else
            As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        int B_col = bx * TILE_SIZE + tx;

        if (B_row < m && B_col < n)
            Bs[ty][tx] = B[B_row * n + B_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            // C[row][col] += A^T[row][i] * B[i][col]
            //              = A[i][row] * B[i][col]
            // In shared mem:
            // As loaded A[t*TILE + ty][by*TILE + tx]
            // We want A[t*TILE + i][row].
            // 'row' corresponds to by*TILE + ty.
            // So we need As[i][ty].
            
            // Bs loaded B[t*TILE + ty][bx*TILE + tx]
            // We want B[t*TILE + i][col].
            // 'col' corresponds to bx*TILE + tx.
            // So we need Bs[i][tx].
            
            sum += As[i][ty] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < k && col < n) C[row * n + col] = sum;
}

void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_transposeA_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void matmul_transposeB_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1]; // Padding for bank conflict

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + tx;
        if (row < m && A_col < k) As[ty][tx] = A[row * k + A_col];
        else As[ty][tx] = 0.0f;

        int B_row = bx * TILE_SIZE + ty; 
        int B_col = t * TILE_SIZE + tx;
        if (B_row < n && B_col < k) Bs[ty][tx] = B[B_row * k + B_col];
        else Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[tx][i];
        }
        __syncthreads();
    }

    if (row < m && col < n) C[row * n + col] = sum;
}

void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);
    matmul_transposeB_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.rows);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void bias_grad_kernel(const float* dY, float* db, int m, int n) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float sdata[TILE_SIZE][TILE_SIZE];

    if (row < m && col < n) sdata[ty][tx] = dY[row * n + col];
    else sdata[ty][tx] = 0.0f;
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (ty < stride) {
            sdata[ty][tx] += sdata[ty + stride][tx];
        }
        __syncthreads();
    }

    if (ty == 0 && col < n) {
        atomicAdd(&db[col], sdata[0][tx]);
    }
}

void computeBiasGradient(const Matrix& dY, Matrix& db) {
    db.zeros();
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dY.cols + blockDim.x - 1) / blockDim.x, (dY.rows + blockDim.y - 1) / blockDim.y);
    bias_grad_kernel<<<gridDim, blockDim>>>(dY.data, db.data, dY.rows, dY.cols);
    CHECK_CUDA(cudaGetLastError());
}


void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX) {
    naive::reluBackward(dY, Z, dX);
}

void updateWeights(Matrix& W, const Matrix& dW, float lr) {
    naive::updateWeights(W, dW, lr);
}

void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss) {
    naive::computeMSEGradient(P, Y, d_loss);
}

void softmaxActivation(const Matrix& Z, Matrix& A) {
    naive::softmaxActivation(Z, A);
}

} // namespace shared

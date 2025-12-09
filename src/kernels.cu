#include "kernels.cuh"
#include <cmath>
#include <cfloat>

#define TILE_SIZE 16

// CUDA Kernel for Matrix Multiplication
// C = A * B
// A: m x k, B: k x n, C: m x n
__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        if (A_row < m && A_col < k)
            As[ty][tx] = A[A_row * k + A_col];
        else
            As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        if (B_row < k && B_col < n)
            Bs[ty][tx] = B[B_row * n + B_col];
        else
            Bs[ty][tx] = 0.0f;

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

// CUDA Kernel for Matrix Multiplication with Bias
// C = A * B + b
__global__ void matmul_with_bias_kernel(const float* A, const float* B, const float* b, float* C, int m, int k, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        if (A_row < m && A_col < k)
            As[ty][tx] = A[A_row * k + A_col];
        else
            As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        if (B_row < k && B_col < n)
            Bs[ty][tx] = B[B_row * n + B_col];
        else
            Bs[ty][tx] = 0.0f;

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
    if (A.cols != B.rows) {
        std::cerr << "Error: Matrix dimensions mismatch for multiplication." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        std::cerr << "Error: Output matrix dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (b.cols != C.cols) {
        std::cerr << "Error: Bias dimension mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);

    matmul_with_bias_kernel<<<gridDim, blockDim>>>(A.data, B.data, b.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for Matrix Multiplication with Bias and ReLU
// C = ReLU(A * B + b)
__global__ void matmul_with_bias_relu_kernel(const float* A, const float* B, const float* b, float* C, int m, int k, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        if (A_row < m && A_col < k)
            As[ty][tx] = A[A_row * k + A_col];
        else
            As[ty][tx] = 0.0f;

        int B_row = t * TILE_SIZE + ty;
        int B_col = col;
        if (B_row < k && B_col < n)
            Bs[ty][tx] = B[B_row * n + B_col];
        else
            Bs[ty][tx] = 0.0f;

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
    if (A.cols != B.rows) {
        std::cerr << "Error: Matrix dimensions mismatch for multiplication." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (C.rows != A.rows || C.cols != B.cols) {
        std::cerr << "Error: Output matrix dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (b.cols != C.cols) {
        std::cerr << "Error: Bias dimension mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);

    matmul_with_bias_relu_kernel<<<gridDim, blockDim>>>(A.data, B.data, b.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

// CUDA Kernel for Adding Bias
// Y: m x n, b: 1 x n
// Each row of Y corresponds to a sample, we add b to each sample.
__global__ void add_bias_kernel(float* Y, const float* b, int m, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Shared memory to cache the bias vector for this tile's columns
    __shared__ float b_shared[TILE_SIZE];

    // Load bias into shared memory
    // Only the first row of threads in the block needs to load it
    if (ty == 0) {
        if (col < n) {
            b_shared[tx] = b[col];
        } else {
            b_shared[tx] = 0.0f;
        }
    }
    __syncthreads();

    if (row < m && col < n) {
        Y[row * n + col] += b_shared[tx];
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
            if (Z[row * n + i] > max_val) max_val = Z[row * n + i];
        }

        // 2. Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i) {
            float val = expf(Z[row * n + i] - max_val);
            A[row * n + i] = val; // Store temporarily
            sum_exp += val;
        }

        // 3. Normalize
        for (int i = 0; i < n; ++i) {
            A[row * n + i] /= sum_exp;
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

// -----------------------------------------------------------
// 1. Matrix Multiplication with Transpose A: C = A^T * B
// Used for: dW = X^T * dZ
// Dimensions: A is (m x k), B is (m x n), C is (k x n)
// Note: Inner dimension for mult is 'm' (rows of A, rows of B)
// -----------------------------------------------------------
#define TILE_SIZE 16

__global__ void matmul_transposeA_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
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

        for (int k_idx = 0; k_idx < TILE_SIZE; ++k_idx) {
            sum += As[k_idx][ty] * Bs[k_idx][tx];
        }

        __syncthreads();
    }

    if (row < k && col < n) {
        C[row * n + col] = sum;
    }
}

void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C) {
    // A: m x k (we treat as k x m), B: m x n
    // Output C: k x n
    // Validation: A.rows must match B.rows for (A^T * B) to work
    if (A.rows != B.rows) {
        std::cerr << "Error: Dimension mismatch for TransposeA Mult." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Grid dimensions based on Output C (k x n)
    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);

    matmul_transposeA_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.cols);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------
// 2. Matrix Multiplication with Transpose B: C = A * B^T
// -----------------------------------------------------------
// 2. Matrix Multiplication with Transpose B: C = A * B^T
// Used for: d_input = dZ * W^T
// Dimensions: A is (m x k), B is (n x k), C is (m x n)
// Note: Inner dimension is 'k' (cols of A, cols of B)
// -----------------------------------------------------------
__global__ void matmul_transposeB_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    // Pad shared memory to avoid bank conflicts when accessing columns
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A: row `row`, col `t*TILE + tx`
        int A_row = row;
        int A_col = t * TILE_SIZE + tx;
        if (A_row < m && A_col < k)
            As[ty][tx] = A[A_row * k + A_col];
        else
            As[ty][tx] = 0.0f;

        // Load B: row `bx*TILE + ty` (part of col of C), col `t*TILE + tx` (part of k)
        // We want B[col][k].
        // We load B rows corresponding to C cols.
        int B_row = bx * TILE_SIZE + ty; 
        int B_col = t * TILE_SIZE + tx;
        
        // B is n x k.
        if (B_row < n && B_col < k)
             Bs[ty][tx] = B[B_row * k + B_col];
        else
             Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            // sum += A[row][k] * B[col][k]
            // A[row][k] is As[ty][i]
            // B[col][k] is Bs[tx][i] (since we loaded B_row corresponding to ty, but we need B_row corresponding to tx)
            sum += As[ty][i] * Bs[tx][i];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C) {
    // A: m x k, B: n x k (treated as k x n)
    // Output C: m x n
    if (A.cols != B.cols) {
        std::cerr << "Error: Dimension mismatch for TransposeB Mult." << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((C.cols + blockDim.x - 1) / blockDim.x, (C.rows + blockDim.y - 1) / blockDim.y);

    matmul_transposeB_kernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.rows, A.cols, B.rows);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------
// 3. Bias Gradient: db = sum(dZ, axis=0)
// Collapses batch dimension. 
// Optimized using Shared Memory Reduction to reduce atomicAdd contention.
// -----------------------------------------------------------

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void bias_grad_kernel(const float* dZ, float* db, int m, int n) {
    // Use 32x32 tile (1024 threads)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * 32 + tx;
    int row = by * 32 + ty;

    // Pad shared memory to avoid bank conflicts
    __shared__ float sdata[32][33];

    // 1. Load data into shared memory (Coalesced)
    if (row < m && col < n) {
        sdata[ty][tx] = dZ[row * n + col];
    } else {
        sdata[ty][tx] = 0.0f;
    }
    __syncthreads();

    // 2. Transpose and Reduce
    // We want to sum along columns (reduce rows).
    // Thread (tx, ty) reads sdata[tx][ty].
    // tx becomes the row index (0..31), ty becomes the column index (0..31).
    // Warp `ty` handles column `ty`.
    
    float val = sdata[tx][ty];
    
    // Sum within the warp (summing over tx, which corresponds to rows)
    val = warpReduceSum(val);

    // 3. Atomic Add
    // Only the first thread in the warp (tx=0) writes the result
    int out_col = bx * 32 + ty;
    if (tx == 0 && out_col < n) {
        atomicAdd(&db[out_col], val);
    }
}

void computeBiasGradient(const Matrix& dY, Matrix& db) {
    // Initialize db to zero first!
    db.zeros();

    dim3 blockDim(32, 32);
    dim3 gridDim((dY.cols + blockDim.x - 1) / blockDim.x, (dY.rows + blockDim.y - 1) / blockDim.y);
    
    bias_grad_kernel<<<gridDim, blockDim>>>(dY.data, db.data, dY.rows, dY.cols);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------
// 4. ReLU Backward: dX = dY * (Z > 0 ? 1 : 0)
// Element-wise multiplication with derivative
// -----------------------------------------------------------
__global__ void relu_backward_kernel(const float* dY, const float* Z, float* dX, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dX[idx] = (Z[idx] > 0.0f) ? dY[idx] : 0.0f;
    }
}

void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX) {
    int size = dY.rows * dY.cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    relu_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(dY.data, Z.data, dX.data, size);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------
// 5. Update Weights: W = W - lr * dW
// -----------------------------------------------------------
__global__ void update_weights_kernel(float* W, const float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}

void updateWeights(Matrix& W, const Matrix& dW, float lr) {
    int size = W.rows * W.cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    update_weights_kernel<<<blocksPerGrid, threadsPerBlock>>>(W.data, dW.data, lr, size);
    CHECK_CUDA(cudaGetLastError());
}

// -----------------------------------------------------------
// 6. MSE Gradient: d_loss = Prediction - Target
// -----------------------------------------------------------
__global__ void mse_gradient_kernel(const float* P, const float* Y, float* d_loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Derivative of MSE (ignoring 2/N scaling for simplicity)
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

// -----------------------------------------------------------
// Softmax Activation (Numerically Stable)
// -----------------------------------------------------------
/*__global__ void softmax_kernel(const float* Z, float* A, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        // 1. Find max for stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < n; ++i) {
            if (Z[row * n + i] > max_val) max_val = Z[row * n + i];
        }

        // 2. Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i) {
            float val = expf(Z[row * n + i] - max_val);
            A[row * n + i] = val; // Store temporarily
            sum_exp += val;
        }

        // 3. Normalize
        for (int i = 0; i < n; ++i) {
            A[row * n + i] /= sum_exp;
        }
    }
}*/

// void softmaxActivation(const Matrix& Z, Matrix& A) {
//     int threadsPerBlock = 256;
//     // One thread per ROW (sample), assuming n is small (10 classes)
//     int blocksPerGrid = (Z.rows + threadsPerBlock - 1) / threadsPerBlock;
//     softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(Z.data, A.data, Z.rows, Z.cols);
//     CHECK_CUDA(cudaGetLastError());
// }

// -----------------------------------------------------------
// Cross Entropy Gradient: dZ = P - Y
// If we use Softmax + Cross Entropy, the gradient simplifies to just (Pred - Target)
// We can reuse the "mse_gradient_kernel" logic or rename it for clarity.
// -----------------------------------------------------------
// (You can reuse the kernel you added for MSE, mathematically it's P - Y for both!)

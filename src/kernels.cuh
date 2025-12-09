#pragma once
#include "utils.cuh"

// Initialize matrix with random values (on host then copy, or kernel)
// For simplicity in primitives, we assume data is already loaded via Matrix struct utils.

// Matrix Multiplication: C = A * B
// A: m x k, B: k x n, C: m x n
void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);

// Matrix Multiplication with Bias: C = A * B + b
void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);

// Matrix Multiplication with Bias and ReLU: C = ReLU(A * B + b)
void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);

// Add Bias: Y = Y + b
// Y: m x n, b: 1 x n (broadcasted over rows) or m x 1? Usually bias is per output neuron.
// If Y is (batch_size, output_features), b is (output_features).
void addBias(Matrix& Y, const Matrix& b);

// ReLU Activation: A = max(0, Z)
void reluActivation(const Matrix& Z, Matrix& A);

// Softmax Activation (for output layer usually)
// Applied row-wise
void softmaxActivation(const Matrix& Z, Matrix& A);

// Computes C = A^T * B
void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C);

// Computes C = A * B^T
void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C);

// Computes gradient of bias (sums rows of dY)
void computeBiasGradient(const Matrix& dY, Matrix& db);

// Computes backprop for ReLU: dX = dY * (Z > 0)
void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX);

// Update: W = W - lr * dW
void updateWeights(Matrix& W, const Matrix& dW, float lr);

// Computes gradient for MSE: d_loss = P - Y
void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss);
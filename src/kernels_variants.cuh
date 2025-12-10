#pragma once
#include "utils.cuh"

enum class KernelMode { NAIVE, SHARED, FUSED, WARP };

namespace naive {
    void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void addBias(Matrix& Y, const Matrix& b);
    void reluActivation(const Matrix& Z, Matrix& A);
    void softmaxActivation(const Matrix& Z, Matrix& A);
    void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C);
    void computeBiasGradient(const Matrix& dY, Matrix& db);
    void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX);
    void updateWeights(Matrix& W, const Matrix& dW, float lr);
    void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss);
}

namespace shared {
    void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void addBias(Matrix& Y, const Matrix& b);
    void reluActivation(const Matrix& Z, Matrix& A);
    void softmaxActivation(const Matrix& Z, Matrix& A);
    void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C);
    void computeBiasGradient(const Matrix& dY, Matrix& db);
    void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX);
    void updateWeights(Matrix& W, const Matrix& dW, float lr);
    void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss);
}

namespace fused {
    void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void addBias(Matrix& Y, const Matrix& b);
    void reluActivation(const Matrix& Z, Matrix& A);
    void softmaxActivation(const Matrix& Z, Matrix& A);
    void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C);
    void computeBiasGradient(const Matrix& dY, Matrix& db);
    void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX);
    void updateWeights(Matrix& W, const Matrix& dW, float lr);
    void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss);
}

namespace warp_opt {
    void matrixMultiply(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyWithBias(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void matrixMultiplyWithBiasAndReLU(const Matrix& A, const Matrix& B, const Matrix& b, Matrix& C);
    void addBias(Matrix& Y, const Matrix& b);
    void reluActivation(const Matrix& Z, Matrix& A);
    void softmaxActivation(const Matrix& Z, Matrix& A);
    void matrixMultiplyTransposeA(const Matrix& A, const Matrix& B, Matrix& C);
    void matrixMultiplyTransposeB(const Matrix& A, const Matrix& B, Matrix& C);
    void computeBiasGradient(const Matrix& dY, Matrix& db);
    void reluBackward(const Matrix& dY, const Matrix& Z, Matrix& dX);
    void updateWeights(Matrix& W, const Matrix& dW, float lr);
    void computeMSEGradient(const Matrix& P, const Matrix& Y, Matrix& d_loss);
}

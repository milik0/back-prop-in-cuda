#pragma once
#include "utils.cuh"
#include <vector>
#include <cublas_v2.h>

struct Layer {
    Matrix W;
    Matrix b;
    Matrix Z; // Linear output: Z = X * W + b
    Matrix A; // Activation output: A = Activation(Z)
    
    int input_dim;
    int output_dim;

    void initialize(int in_dim, int out_dim);
    void free();
};
class MLP {
public:
    std::vector<Layer> layers;
    cublasHandle_t handle;

    MLP(const std::vector<int>& layer_sizes);
    MLP(const std::vector<int>& layer_sizes);
    ~MLP();

    void forward(const Matrix& X);
    void backpropagation(const Matrix& X, const Matrix& Y, float learning_rate);
    
    Matrix& getOutput();
};

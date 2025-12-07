#include "mlp.cuh"
#include "kernels.cuh"
#include <random>
#include <iostream>

void Layer::initialize(int in_dim, int out_dim) {
    input_dim = in_dim;
    output_dim = out_dim;
    W.allocate(in_dim, out_dim);
    b.allocate(1, out_dim);
    
    Z.data = nullptr;
    A.data = nullptr;
    Z.rows = 0; Z.cols = 0;
    A.rows = 0; A.cols = 0;

    // Initialize W and b with random values
    std::vector<float> h_W(in_dim * out_dim);
    std::vector<float> h_b(out_dim);
    
    // Simple random init
    for(auto& v : h_W) v = ((float)rand() / RAND_MAX) * 0.1f; // Small random values
    for(auto& v : h_b) v = 0.0f;

    W.copyFromHost(h_W);
    b.copyFromHost(h_b);
}

void Layer::free() {
    W.free();
    b.free();
    if (Z.data) Z.free();
    if (A.data) A.free();
}

MLP::MLP(const std::vector<int>& layer_sizes) {
    cublasCreate(&handle);
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        Layer l;
        l.initialize(layer_sizes[i], layer_sizes[i+1]);
        layers.push_back(l);
    }
}

MLP::~MLP() {
    for (auto& l : layers) {
        l.free();
    }
    cublasDestroy(handle);
}

void MLP::forward(const Matrix& X) {
    const Matrix* input = &X;

    for (size_t i = 0; i < layers.size(); ++i) {
        Layer& layer = layers[i];
        
        // Reallocate Z and A if batch size changes
        if (layer.Z.rows != input->rows || layer.Z.cols != layer.output_dim) {
            if (layer.Z.data) layer.Z.free();
            if (layer.A.data) layer.A.free();
            layer.Z.allocate(input->rows, layer.output_dim);
            layer.A.allocate(input->rows, layer.output_dim);
        }
        // 1. Linear: Z = Input * W
        // matrixMultiply(*input, layer.W, layer.Z);
        float alpha = 1.0f;
        float beta = 0.0f;
        // cuBLAS uses column-major storage.
        // We want C = A * B (row-major).
        // In column-major, this is equivalent to C^T = B^T * A^T.
        // A (m x k), B (k x n), C (m x n)
        // We pass B as A_cublas (n x k), A as B_cublas (k x m), C as C_cublas (n x m)
        int m = input->rows;
        int k = input->cols;
        int n = layer.W.cols;
        
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k, 
                    &alpha, 
                    layer.W.data, n, 
                    input->data, k, 
                    &beta, 
                    layer.Z.data, n);

        // 2. Bias: Z = Z + b
        // 2. Bias: Z = Z + b
        addBias(layer.Z, layer.b);

        // 3. Activation
        if (i == layers.size() - 1) {
            // Last layer: Softmax
            softmaxActivation(layer.Z, layer.A);
        } else {
            // Hidden layers: ReLU
            reluActivation(layer.Z, layer.A);
        }

        // Output of this layer is input to next
        input = &layer.A;
    }
}

void MLP::backpropagation(const Matrix& X, const Matrix& Y, float learning_rate) {
    // To be implemented
    std::cout << "Backpropagation not yet implemented." << std::endl;
}

Matrix& MLP::getOutput() {
    return layers.back().A;
}

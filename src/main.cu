#include "utils.cuh"
#include "kernels.cuh"
#include <iostream>
#include <vector>

void printMatrix(const std::vector<float>& data, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Dimensions
    int batch_size = 2;
    int input_features = 3;
    int output_features = 2;

    // Host Data
    std::vector<float> h_X = {1.0f, 2.0f, 3.0f,  // Sample 1
                              4.0f, 5.0f, 6.0f}; // Sample 2
    
    std::vector<float> h_W = {0.1f, 0.2f,
                              0.3f, 0.4f,
                              0.5f, 0.6f}; // 3x2 Matrix
    
    std::vector<float> h_b = {0.1f, 0.2f}; // Bias for 2 outputs

    // Device Matrices
    Matrix d_X, d_W, d_b, d_Z, d_A;
    d_X.allocate(batch_size, input_features);
    d_W.allocate(input_features, output_features);
    d_b.allocate(1, output_features);
    d_Z.allocate(batch_size, output_features);
    d_A.allocate(batch_size, output_features);

    // Copy to Device
    d_X.copyFromHost(h_X);
    d_W.copyFromHost(h_W);
    d_b.copyFromHost(h_b);

    // 1. Linear Forward: Z = X * W
    matrixMultiply(d_X, d_W, d_Z);
    
    // 2. Add Bias: Z = Z + b
    addBias(d_Z, d_b);

    // Copy back to check Linear result
    std::vector<float> h_Z;
    d_Z.copyToHost(h_Z);
    printMatrix(h_Z, batch_size, output_features, "Linear Output (Z)");

    // 3. Activation: A = ReLU(Z)
    // Let's modify Z to have some negative values to test ReLU
    // But for now, let's just run it.
    reluActivation(d_Z, d_A);

    // Copy back
    std::vector<float> h_A;
    d_A.copyToHost(h_A);
    printMatrix(h_A, batch_size, output_features, "ReLU Output (A)");

    // 4. Softmax
    softmaxActivation(d_Z, d_A);
    d_A.copyToHost(h_A);
    printMatrix(h_A, batch_size, output_features, "Softmax Output (A)");

    // Cleanup
    d_X.free();
    d_W.free();
    d_b.free();
    d_Z.free();
    d_A.free();

    return 0;
}

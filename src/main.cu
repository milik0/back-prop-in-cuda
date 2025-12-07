#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
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

    // Define MLP structure
    // Input: 3 features
    // Output: 2 classes
    // Let's add a hidden layer to make it a "Multi-Layer" Perceptron
    // Structure: 3 -> 4 -> 2
    std::vector<int> structure = {input_features, 4, output_features};
    MLP mlp(structure);

    std::cout << "Created MLP with structure: ";
    for (int s : structure) std::cout << s << " ";
    std::cout << "\n\n";

    // Host Data
    std::vector<float> h_X = {1.0f, 2.0f, 3.0f,  // Sample 1
                              4.0f, 5.0f, 6.0f}; // Sample 2
    
    // Device Matrices
    Matrix d_X;
    d_X.allocate(batch_size, input_features);
    d_X.copyFromHost(h_X);

    // Forward Pass
    std::cout << "Running Forward Pass..." << std::endl;
    mlp.forward(d_X);

    // Get Output
    Matrix& d_Output = mlp.getOutput();
    
    // Copy back
    std::vector<float> h_Output;
    d_Output.copyToHost(h_Output);
    printMatrix(h_Output, batch_size, output_features, "MLP Output");

    // Cleanup
    d_X.free();
    // MLP destructor handles layer cleanup

    return 0;
}

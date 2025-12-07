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
    // 1. Setup Data
    int batch_size = 2;
    Matrix d_X; d_X.allocate(batch_size, 3);
    // ... fill d_X with data ...

    // 2. Build MLP
    MLP model;
    
    // Layer 1: 3 inputs -> 4 hidden
    Linear* fc1 = new Linear(3, 4); 
    // Initialize fc1->W and fc1->b here with random values!
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2: 4 hidden -> 2 outputs
    Linear* fc2 = new Linear(4, 2);
    // Initialize fc2->W and fc2->b here!
    model.add(fc2);

    // 3. Training Loop (Simplified)
    float learning_rate = 0.01f;
    
    // Forward
    Matrix prediction = model.forward(d_X);

    // Compute Loss Gradient (d_loss) manually for now
    // e.g., if Loss = MSE, then d_loss = 2 * (prediction - target)
    Matrix d_loss; 
    d_loss.allocate(batch_size, 2); 
    // ... fill d_loss ...

    // Backward
    model.backward(d_loss, learning_rate);

    // Cleanup done by MLP destructor
    d_X.free();
    d_loss.free();
    
    return 0;
}
#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <random>

// Helper to print a Matrix object
void printMat(Matrix& m, const char* name) {
    std::vector<float> host_data;
    m.copyToHost(host_data); // Copy from GPU to CPU
    
    std::cout << name << " (" << m.rows << "x" << m.cols << "):\n";
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            std::cout << host_data[i * m.cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------\n";
}

// Helper to fill a Matrix with random values
void randomize(Matrix& m, float min = -0.5f, float max = 0.5f) {
    std::vector<float> host_data(m.rows * m.cols);
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
    m.copyFromHost(host_data); // Copy from CPU to GPU
}

int main() {
    srand(1337); // Fixed seed for reproducibility

    // 1. Setup Data
    int batch_size = 2;
    int input_dim = 3;
    int hidden_dim = 4;
    int output_dim = 2;

    std::cout << "=== Setting up Data ===\n";
    Matrix d_X; 
    d_X.allocate(batch_size, input_dim);
    randomize(d_X); // Fill X with random numbers
    printMat(d_X, "Input X");

    // 2. Build MLP
    std::cout << "=== Building MLP ===\n";
    MLP model;
    
    // Layer 1
    Linear* fc1 = new Linear(input_dim, hidden_dim); 
    randomize(fc1->W); // Important: Init weights!
    randomize(fc1->b); 
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2
    Linear* fc2 = new Linear(hidden_dim, output_dim);
    randomize(fc2->W); // Important: Init weights!
    randomize(fc2->b);
    model.add(fc2);

    // 3. Forward Pass
    std::cout << "=== Forward Pass ===\n";
    Matrix prediction = model.forward(d_X);
    printMat(prediction, "Prediction (Output)");

    // 4. Backward Pass Setup
    // Create a fake gradient (d_loss) to simulate a loss function
    Matrix d_loss; 
    d_loss.allocate(batch_size, output_dim);
    randomize(d_loss); 
    printMat(d_loss, "Fake Loss Gradient (d_loss)");

    // 5. Run Backward
    std::cout << "=== Backward Pass ===\n";
    float learning_rate = 0.1f;
    model.backward(d_loss, learning_rate);

    // Verify Weights Updated
    // We print the weights of the last layer to see if they changed
    printMat(fc2->W, "Updated Weights (Layer 2)");

    // Cleanup
    d_X.free();
    d_loss.free();
    // model destructor cleans up layers
    
    return 0;
}
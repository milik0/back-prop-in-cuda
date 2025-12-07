#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::setw

// Helper: Initialize random matrix
void randomize(Matrix& m, float min = -0.5f, float max = 0.5f) {
    std::vector<float> host_data(m.rows * m.cols);
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
    m.copyFromHost(host_data);
}

// Helper: Calculate MSE on CPU (for logging only)
float calculate_mse_cpu(Matrix& preds, Matrix& target) {
    std::vector<float> h_p, h_t;
    preds.copyToHost(h_p);
    target.copyToHost(h_t);
    
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < h_p.size(); ++i) {
        float diff = h_p[i] - h_t[i];
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / h_p.size();
}

// ==========================================
// The Training Loop
// ==========================================
void train(MLP& model, Matrix& X, Matrix& Y, int epochs, float lr) {
    Matrix d_loss;
    d_loss.allocate(Y.rows, Y.cols);

    std::cout << "Starting Training (" << epochs << " epochs)...\n";

    for (int i = 0; i < epochs; ++i) {
        // 1. Forward
        Matrix preds = model.forward(X);

        // 2. Log Progress (every 100 epochs)
        if (i % 100 == 0) {
            float loss = calculate_mse_cpu(preds, Y);
            std::cout << "Epoch " << std::setw(4) << i << " | MSE Loss: " << loss << std::endl;
        }

        // 3. Compute Gradient (d_loss = Preds - Y)
        computeMSEGradient(preds, Y, d_loss);

        // 4. Backward & Update
        model.backward(d_loss, lr);
    }
    
    // Final Loss
    Matrix final_preds = model.forward(X);
    std::cout << "Final Epoch | MSE Loss: " << calculate_mse_cpu(final_preds, Y) << std::endl;

    d_loss.free();
}

int main() {
    srand(1337);

    // --- 1. Data Setup ---
    int batch_size = 4; // Using 4 samples
    int input_dim = 3;
    int hidden_dim = 16;
    int output_dim = 2;

    Matrix d_X, d_Y;
    d_X.allocate(batch_size, input_dim);
    d_Y.allocate(batch_size, output_dim);
    
    randomize(d_X); // Random Inputs
    randomize(d_Y); // Random Targets (we want the network to memorize these)

    // --- 2. Model Setup ---
    MLP model;
    
    // Layer 1
    Linear* fc1 = new Linear(input_dim, hidden_dim);
    randomize(fc1->W); randomize(fc1->b);
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2
    Linear* fc2 = new Linear(hidden_dim, output_dim);
    randomize(fc2->W); randomize(fc2->b);
    model.add(fc2);

    // --- 3. Run Training ---
    // High learning rate because we haven't normalized inputs
    train(model, d_X, d_Y, 1000, 0.01f); 

    // --- 4. Cleanup ---
    d_X.free();
    d_Y.free();

    return 0;
}
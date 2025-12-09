#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

// Helper to calculate MSE on CPU (for logging)
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

// Helper to init weights (Xavier/Random)
void randomize(Matrix& m, float min = -0.5f, float max = 0.5f) {
    std::vector<float> host_data(m.rows * m.cols);
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
    m.copyFromHost(host_data);
}

float train(MLP& model, Matrix& X, Matrix& Y, int epochs, float lr) {
    Matrix d_loss;
    d_loss.allocate(Y.rows, Y.cols);

    std::cout << "Training on XOR (" << epochs << " epochs)...\n";
    
    // Warmup: run a few iterations to initialize everything
    for (int i = 0; i < 10; ++i) {
        Matrix preds = model.forward(X);
        computeMSEGradient(preds, Y, d_loss);
        model.backward(d_loss, lr);
    }
    
    // Synchronize GPU before timing
    cudaDeviceSynchronize();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < epochs; ++i) {
        // Forward
        Matrix preds = model.forward(X);

        // Log every 1000 steps
        if (i % 1000 == 0) {
            float loss = calculate_mse_cpu(preds, Y);
            std::cout << "Epoch " << std::setw(5) << i << " | Loss: " << loss << std::endl;
        }

        // Backward
        computeMSEGradient(preds, Y, d_loss);
        model.backward(d_loss, lr);
    }
    
    // Synchronize GPU after training
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    float training_time = elapsed.count();
    
    d_loss.free();
    return training_time;
}

int main() {
    srand(1337); 

    // ==========================================
    // 1. Define XOR Dataset
    // ==========================================
    int batch_size = 4;
    int input_dim = 2;
    int hidden_dim = 8; // Enough neurons to solve XOR
    int output_dim = 1; // Output probability (0 or 1)

    // Inputs: (0,0), (0,1), (1,0), (1,1)
    std::vector<float> h_X = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    // Targets: 0, 1, 1, 0
    std::vector<float> h_Y = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    Matrix d_X, d_Y;
    d_X.allocate(batch_size, input_dim);
    d_Y.allocate(batch_size, output_dim);
    
    // Copy data to GPU
    d_X.copyFromHost(h_X);
    d_Y.copyFromHost(h_Y);

    // ==========================================
    // 2. Build Model
    // ==========================================
    MLP model;
    
    // Layer 1: 2 -> 8 (ReLU)
    Linear* fc1 = new Linear(input_dim, hidden_dim);
    randomize(fc1->W); randomize(fc1->b);
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2: 8 -> 1 (Linear output for now)
    // Note: Usually we'd use Sigmoid here for 0-1, but linear works with MSE for this demo
    Linear* fc2 = new Linear(hidden_dim, output_dim);
    randomize(fc2->W); randomize(fc2->b);
    model.add(fc2);

    // ==========================================
    // 3. Train
    // ==========================================
    float training_time = train(model, d_X, d_Y, 10000, 0.1f);
    std::cout << "\nTraining Time: " << training_time << " seconds\n";

    // ==========================================
    // 4. Verify Results
    // ==========================================
    std::cout << "\n=== Final Predictions ===\n";
    Matrix final_preds = model.forward(d_X);
    
    std::vector<float> res;
    final_preds.copyToHost(res);
    // Add this header at the top
    std::cout << std::fixed << std::setprecision(9); // Force 9 decimal places

    std::cout << "Input (0, 0) -> Pred: " << res[0] << " (Target: 0)\n";
    std::cout << "Input (0, 1) -> Pred: " << res[1] << " (Target: 1)\n";
    std::cout << "Input (1, 0) -> Pred: " << res[2] << " (Target: 1)\n";
    std::cout << "Input (1, 1) -> Pred: " << res[3] << " (Target: 0)\n";

    // Cleanup
    d_X.free(); d_Y.free();

    return 0;
}
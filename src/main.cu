#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include "mnist.cuh" // The new file
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm> // for std::shuffle

// Helper: Calculate Accuracy on CPU
float calculate_accuracy(Matrix& preds, Matrix& targets) {
    std::vector<float> h_p, h_t;
    preds.copyToHost(h_p);
    targets.copyToHost(h_t);
    
    int hits = 0;
    int batch_size = preds.rows;
    int cols = preds.cols; // 10

    for (int i = 0; i < batch_size; ++i) {
        // Find Argmax Prediction
        int max_p_idx = 0;
        float max_p_val = h_p[i * cols];
        for (int j = 1; j < cols; ++j) {
            if (h_p[i * cols + j] > max_p_val) {
                max_p_val = h_p[i * cols + j];
                max_p_idx = j;
            }
        }

        // Find Argmax Target
        int max_t_idx = 0;
        float max_t_val = h_t[i * cols];
        for (int j = 1; j < cols; ++j) {
            if (h_t[i * cols + j] > max_t_val) {
                max_t_val = h_t[i * cols + j];
                max_t_idx = j;
            }
        }

        if (max_p_idx == max_t_idx) hits++;
    }
    return (float)hits / batch_size;
}

// Helper to init weights
void init_xavier(Matrix& m) {
    float scale = sqrt(2.0f / m.rows);
    std::vector<float> host_data(m.rows * m.cols);
    for (size_t i = 0; i < host_data.size(); ++i) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // 0 to 1
        host_data[i] = (r * 2.0f - 1.0f) * scale;
    }
    m.copyFromHost(host_data);
}

int main() {
    srand(1337); 

    // 1. Load MNIST Data
    // Ensure you have these files in a "data" folder!
    std::cout << "Loading MNIST Data...\n";
    Matrix full_X, full_Y;
    int N_SAMPLES = 60000; // Load all training samples
    loadMNIST("/root/khaled-sans-bapt/back-prop-in-cuda/data/train-images-idx3-ubyte", "/root/khaled-sans-bapt/back-prop-in-cuda/data/train-labels-idx1-ubyte", full_X, full_Y, N_SAMPLES);

    // 2. Build Model (784 -> 256 -> 10)
    MLP model;
    
    // Layer 1
    Linear* fc1 = new Linear(784, 256);
    init_xavier(fc1->W); fc1->b.zeros();
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2 (Output)
    Linear* fc2 = new Linear(256, 10);
    init_xavier(fc2->W); fc2->b.zeros();
    model.add(fc2);

    // Final Activation & Loss
    model.add(new SoftmaxCrossEntropy());

    // 3. Mini-Batch Training Loop
    int batch_size = 64;
    int epochs = 5;
    float learning_rate = 0.01f;
    int num_batches = N_SAMPLES / batch_size;

    Matrix batch_X, batch_Y;
    batch_X.allocate(batch_size, 784);
    batch_Y.allocate(batch_size, 10);

    std::cout << "Starting Training (" << epochs << " epochs, batch size " << batch_size << ")...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_acc = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // Slice Batch (Copy from full dataset to batch matrix)
            // Note: This copying is slow (D->H->D) usually, but for simplicity we do it here.
            // kernel "slice" copy would be faster.
            CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 784, 
                                  batch_size * 784 * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * 10, 
                                  batch_size * 10 * sizeof(float), cudaMemcpyDeviceToDevice));

            // Forward
            Matrix preds = model.forward(batch_X);

            // Backward
            // For SoftmaxCrossEntropy, the "backward" takes the LABELS, not d_loss
            model.backward(batch_Y, learning_rate);

            // Logging
            if (b % 100 == 0) {
                 float acc = calculate_accuracy(preds, batch_Y);
                 // std::cout << "Epoch " << epoch << " Batch " << b << " Acc: " << acc << "\r" << std::flush;
            }
            total_acc += calculate_accuracy(preds, batch_Y);
        }
        std::cout << "Epoch " << epoch << " | Avg Accuracy: " << (total_acc / num_batches) * 100.0f << "%" << std::endl;
    }

    std::cout << "\n=== Final Evaluation on Test Set ===\n";
    
    // 1. Load Test Data
    Matrix test_X, test_Y;
    int N_TEST_SAMPLES = 10000;
    loadMNIST("/root/khaled-sans-bapt/back-prop-in-cuda/data/t10k-images-idx3-ubyte", "/root/khaled-sans-bapt/back-prop-in-cuda/data/t10k-labels-idx1-ubyte", test_X, test_Y, N_TEST_SAMPLES);
    // 2. Run Inference (Forward Pass only)
    // We process in one giant batch since 10k fits easily in GPU memory usually.
    // If you run out of memory, split this like the training loop.
    Matrix test_preds = model.forward(test_X);

    // 3. Calculate Accuracy
    float test_acc = calculate_accuracy(test_preds, test_Y);
    std::cout << "Test Set Accuracy: " << test_acc * 100.0f << "%" << std::endl;

    // Cleanup Test Data
    test_X.free(); test_Y.free();
    // Cleanup
    full_X.free(); full_Y.free();
    batch_X.free(); batch_Y.free();

    return 0;
}
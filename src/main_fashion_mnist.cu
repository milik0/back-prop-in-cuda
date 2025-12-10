#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include "mnist.cuh" // Reuse MNIST loader (same format)
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <chrono>

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
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        host_data[i] = (r * 2.0f - 1.0f) * scale;
    }
    m.copyFromHost(host_data);
}

int main(int argc, char** argv) {
    srand(1337); 

    // Get data paths from command line arguments or use defaults
    std::string train_images = (argc > 1) ? argv[1] : "data/fashion_mnist/train-images-idx3-ubyte";
    std::string train_labels = (argc > 2) ? argv[2] : "data/fashion_mnist/train-labels-idx1-ubyte";
    std::string test_images = (argc > 3) ? argv[3] : "data/fashion_mnist/t10k-images-idx3-ubyte";
    std::string test_labels = (argc > 4) ? argv[4] : "data/fashion_mnist/t10k-labels-idx1-ubyte";

    // 1. Load Fashion-MNIST Data
    std::cout << "Loading Fashion-MNIST Data...\n";
    Matrix full_X, full_Y;
    int N_SAMPLES = 60000;
    loadMNIST(train_images, train_labels, full_X, full_Y, N_SAMPLES);

    // 2. Build Model (784 -> 128 -> 10) - Smaller hidden layer for fashion
    MLP model;
    
    // Layer 1
    Linear* fc1 = new Linear(784, 128);
    init_xavier(fc1->W); fc1->b.zeros();
    model.add(fc1);
    
    model.add(new ReLU());

    // Layer 2 (Output)
    Linear* fc2 = new Linear(128, 10);
    init_xavier(fc2->W); fc2->b.zeros();
    model.add(fc2);

    // Final Activation & Loss
    model.add(new SoftmaxCrossEntropy());

    // 3. Mini-Batch Training Loop
    int batch_size = 64;
    int epochs = 10;
    float learning_rate = 0.01f;
    int num_batches = N_SAMPLES / batch_size;

    Matrix batch_X, batch_Y;
    batch_X.allocate(batch_size, 784);
    batch_Y.allocate(batch_size, 10);

    std::cout << "Starting Training (" << epochs << " epochs, batch size " << batch_size << ")...\n";

    // Warmup: run a few batches to initialize everything
    for (int b = 0; b < 10 && b < num_batches; ++b) {
        CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 784, 
                              batch_size * 784 * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * 10, 
                              batch_size * 10 * sizeof(float), cudaMemcpyDeviceToDevice));
        Matrix preds = model.forward(batch_X);
        model.backward(batch_Y, learning_rate);
    }
    
    // Synchronize GPU before timing
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_acc = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // Slice Batch
            CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 784, 
                                  batch_size * 784 * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * 10, 
                                  batch_size * 10 * sizeof(float), cudaMemcpyDeviceToDevice));

            // Forward
            Matrix preds = model.forward(batch_X);

            // Backward
            model.backward(batch_Y, learning_rate);

            total_acc += calculate_accuracy(preds, batch_Y);
        }
        std::cout << "Epoch " << epoch << " | Avg Accuracy: " << (total_acc / num_batches) * 100.0f << "%" << std::endl;
    }
    
    // Synchronize GPU after training
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nTraining Time: " << elapsed.count() << " seconds\n";

    std::cout << "\n=== Final Evaluation on Test Set ===\n";
    
    // 1. Load Test Data
    Matrix test_X, test_Y;
    int N_TEST_SAMPLES = 10000;
    loadMNIST(test_images, test_labels, test_X, test_Y, N_TEST_SAMPLES);

    // 2. Run Inference
    Matrix test_preds = model.forward(test_X);

    // 3. Calculate Accuracy
    float test_acc = calculate_accuracy(test_preds, test_Y);
    std::cout << "Test Set Accuracy: " << test_acc * 100.0f << "%" << std::endl;

    // Cleanup
    test_X.free(); test_Y.free();
    full_X.free(); full_Y.free();
    batch_X.free(); batch_Y.free();

    return 0;
}

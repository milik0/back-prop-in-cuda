#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

// Helper: Calculate Accuracy on CPU (Binary Classification)
float calculate_accuracy(Matrix& preds, Matrix& targets) {
    std::vector<float> h_p, h_t;
    preds.copyToHost(h_p);
    targets.copyToHost(h_t);
    
    int hits = 0;
    int batch_size = preds.rows;

    for (int i = 0; i < batch_size; ++i) {
        // Binary classification: pred > 0.5 => class 1, else class 0
        int pred_class = (h_p[i] > 0.5f) ? 1 : 0;
        int true_class = (int)(h_t[i] + 0.5f); // Round to nearest int
        
        if (pred_class == true_class) hits++;
    }
    return (float)hits / batch_size;
}

// Helper to load Breast Cancer Wisconsin dataset from CSV
void loadBreastCancer(const std::string& csv_path, Matrix& X, Matrix& Y, int& num_samples) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << csv_path << std::endl;
        exit(1);
    }

    std::vector<std::vector<float>> data;
    std::vector<float> labels;
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        
        // Skip ID column
        std::getline(ss, value, ',');
        
        // Read diagnosis (M=1, B=0)
        std::getline(ss, value, ',');
        float label = (value == "M") ? 1.0f : 0.0f;
        labels.push_back(label);
        
        // Read features
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }
    
    num_samples = data.size();
    int num_features = data[0].size();
    
    std::cout << "Loaded Breast Cancer data: " << num_samples << " samples, " 
              << num_features << " features\n";
    
    // Normalize features (simple min-max normalization)
    std::vector<float> min_vals(num_features, 1e9);
    std::vector<float> max_vals(num_features, -1e9);
    
    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            min_vals[j] = std::min(min_vals[j], row[j]);
            max_vals[j] = std::max(max_vals[j], row[j]);
        }
    }
    
    std::vector<float> h_X(num_samples * num_features);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            float normalized = (data[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8f);
            h_X[i * num_features + j] = normalized;
        }
    }
    
    // Copy to GPU
    X.allocate(num_samples, num_features);
    X.copyFromHost(h_X);
    
    Y.allocate(num_samples, 1);
    Y.copyFromHost(labels);
    
    file.close();
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

    // Get data path from command line or use default
    std::string csv_path = (argc > 1) ? argv[1] : "data/breast_cancer.csv";

    // 1. Load Breast Cancer Data
    std::cout << "Loading Breast Cancer Wisconsin Dataset...\n";
    Matrix full_X, full_Y;
    int num_samples;
    loadBreastCancer(csv_path, full_X, full_Y, num_samples);

    // Split into train/test (80/20)
    int train_size = (int)(num_samples * 0.8);
    int test_size = num_samples - train_size;
    
    // 2. Build Model (30 -> 16 -> 8 -> 1) - Binary classification
    MLP model;
    
    // Layer 1
    Linear* fc1 = new Linear(30, 16);
    init_xavier(fc1->W); fc1->b.zeros();
    model.add(fc1);
    model.add(new ReLU());

    // Layer 2
    Linear* fc2 = new Linear(16, 8);
    init_xavier(fc2->W); fc2->b.zeros();
    model.add(fc2);
    model.add(new ReLU());

    // Output Layer
    Linear* fc3 = new Linear(8, 1);
    init_xavier(fc3->W); fc3->b.zeros();
    model.add(fc3);

    // 3. Training Loop
    int batch_size = 32;
    int epochs = 50;
    float learning_rate = 0.01f;
    int num_batches = train_size / batch_size;

    Matrix batch_X, batch_Y, d_loss;
    batch_X.allocate(batch_size, 30);
    batch_Y.allocate(batch_size, 1);
    d_loss.allocate(batch_size, 1);

    std::cout << "Starting Training (" << epochs << " epochs, batch size " << batch_size << ")...\n";

    // Warmup
    for (int b = 0; b < 10 && b < num_batches; ++b) {
        CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 30,
                              batch_size * 30 * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size,
                              batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
        Matrix preds = model.forward(batch_X);
        computeMSEGradient(preds, batch_Y, d_loss);
        model.backward(d_loss, learning_rate);
    }
    
    // Synchronize and start timing
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_acc = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            // Slice Batch
            CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 30,
                                  batch_size * 30 * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size,
                                  batch_size * sizeof(float), cudaMemcpyDeviceToDevice));

            // Forward
            Matrix preds = model.forward(batch_X);

            // Backward (MSE loss for binary classification)
            computeMSEGradient(preds, batch_Y, d_loss);
            model.backward(d_loss, learning_rate);

            total_acc += calculate_accuracy(preds, batch_Y);
        }
        
        if (epoch % 5 == 0) {
            std::cout << "Epoch " << epoch << " | Avg Accuracy: " 
                      << (total_acc / num_batches) * 100.0f << "%" << std::endl;
        }
    }
    
    // Synchronize and end timing
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nTraining Time: " << elapsed.count() << " seconds\n";

    std::cout << "\n=== Final Evaluation on Test Set ===\n";
    
    // Test on remaining data
    int test_batches = test_size / batch_size;
    float total_test_acc = 0.0f;
    
    for (int b = 0; b < test_batches; ++b) {
        int offset = train_size + b * batch_size;
        CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + offset * 30,
                              batch_size * 30 * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + offset,
                              batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        Matrix preds = model.forward(batch_X);
        total_test_acc += calculate_accuracy(preds, batch_Y);
    }
    
    std::cout << "Test Set Accuracy: " << (total_test_acc / test_batches) * 100.0f << "%" << std::endl;

    // Cleanup
    full_X.free(); full_Y.free();
    batch_X.free(); batch_Y.free();
    d_loss.free();

    return 0;
}

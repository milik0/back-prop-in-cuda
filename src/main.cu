#include "utils.cuh"
#include "kernels_variants.cuh"
#include "mlp.cuh"
#include "mnist.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <fstream>

struct BenchmarkResult {
    std::string name;
    float avg_forward_ms;
    float avg_backward_ms;
    float total_time_s;
};

// Helper: Calculate Accuracy on CPU
float calculate_accuracy(Matrix& preds, Matrix& targets) {
    std::vector<float> h_p, h_t;
    preds.copyToHost(h_p);
    targets.copyToHost(h_t);
    
    int hits = 0;
    int batch_size = preds.rows;
    int cols = preds.cols; // 10

    for (int i = 0; i < batch_size; ++i) {
        int max_p_idx = 0;
        float max_p_val = h_p[i * cols];
        for (int j = 1; j < cols; ++j) {
            if (h_p[i * cols + j] > max_p_val) {
                max_p_val = h_p[i * cols + j];
                max_p_idx = j;
            }
        }

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

BenchmarkResult run_benchmark(KernelMode mode, std::string name, Matrix& full_X, Matrix& full_Y, int N_SAMPLES) {
    std::cout << "\n========================================\n";
    std::cout << "Running Benchmark: " << name << "\n";
    std::cout << "========================================\n";

    srand(1337); 

    MLP model;
    
    // Layer 1 (Fused Linear + ReLU)
    LinearReLU* fc1 = new LinearReLU(784, 256, mode);
    init_xavier(fc1->W); fc1->b.zeros();
    model.add(fc1);

    // Layer 2 (Output)
    Linear* fc2 = new Linear(256, 10, mode);
    init_xavier(fc2->W); fc2->b.zeros();
    model.add(fc2);

    // Final Activation & Loss
    model.add(new SoftmaxCrossEntropy(mode));

    int batch_size = 64;
    int epochs = 3; // Reduced for quick benchmarking
    float learning_rate = 0.01f;
    int num_batches = N_SAMPLES / batch_size;

    Matrix batch_X, batch_Y;
    batch_X.allocate(batch_size, 784);
    batch_Y.allocate(batch_size, 10);

    // Warmup
    for (int b = 0; b < 5 && b < num_batches; ++b) {
        CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 784, 
                              batch_size * 784 * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * 10, 
                              batch_size * 10 * sizeof(float), cudaMemcpyDeviceToDevice));
        model.forward(batch_X);
        model.backward(batch_Y, learning_rate);
    }
    cudaDeviceSynchronize();

    auto start_time = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_forward_ms_all = 0.0f;
    float total_backward_ms_all = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_forward_ms = 0.0f;
        float total_backward_ms = 0.0f;

        for (int b = 0; b < num_batches; ++b) {
            CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * 784, 
                                  batch_size * 784 * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * 10, 
                                  batch_size * 10 * sizeof(float), cudaMemcpyDeviceToDevice));

            // Forward
            CHECK_CUDA(cudaEventRecord(start));
            Matrix preds = model.forward(batch_X);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float milliseconds = 0;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            total_forward_ms += milliseconds;

            // Backward
            CHECK_CUDA(cudaEventRecord(start));
            model.backward(batch_Y, learning_rate);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            total_backward_ms += milliseconds;
        }
        total_forward_ms_all += total_forward_ms;
        total_backward_ms_all += total_backward_ms;
        
        std::cout << "Epoch " << epoch << " | Avg Fwd: " << total_forward_ms / num_batches << " ms"
                  << " | Avg Bwd: " << total_backward_ms / num_batches << " ms" << std::endl;
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    float avg_fwd = total_forward_ms_all / (epochs * num_batches);
    float avg_bwd = total_backward_ms_all / (epochs * num_batches);
    float total_time = elapsed.count();

    std::cout << "Total Time: " << total_time << " s\n";
    std::cout << "Average Forward per Batch: " << avg_fwd << " ms\n";
    std::cout << "Average Backward per Batch: " << avg_bwd << " ms\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    batch_X.free(); batch_Y.free();

    return {name, avg_fwd, avg_bwd, total_time};
}

void write_results_to_json(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream out(filename);
    out << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        out << "  {\n";
        out << "    \"name\": \"" << results[i].name << "\",\n";
        out << "    \"avg_forward_ms\": " << results[i].avg_forward_ms << ",\n";
        out << "    \"avg_backward_ms\": " << results[i].avg_backward_ms << ",\n";
        out << "    \"total_time_s\": " << results[i].total_time_s << "\n";
        out << "  }" << (i < results.size() - 1 ? "," : "") << "\n";
    }
    out << "]\n";
    out.close();
    std::cout << "Benchmark results written to " << filename << "\n";
}

int main() {
    std::cout << "Loading MNIST Data...\n";
    Matrix full_X, full_Y;
    int N_SAMPLES = 60000;
    loadMNIST("/root/khaled/data/mnist/train-images-idx3-ubyte", "/root/khaled/data/mnist/train-labels-idx1-ubyte", full_X, full_Y, N_SAMPLES);

    std::vector<BenchmarkResult> results;
    results.push_back(run_benchmark(KernelMode::NAIVE, "Naive", full_X, full_Y, N_SAMPLES));
    results.push_back(run_benchmark(KernelMode::SHARED, "Shared Memory", full_X, full_Y, N_SAMPLES));
    results.push_back(run_benchmark(KernelMode::FUSED, "Fused Kernels", full_X, full_Y, N_SAMPLES));
    results.push_back(run_benchmark(KernelMode::WARP, "Warp Reduce", full_X, full_Y, N_SAMPLES));

    write_results_to_json(results, "benchmark_results.json");

    full_X.free(); full_Y.free();
    return 0;
}
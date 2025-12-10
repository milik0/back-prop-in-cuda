#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

#include "utils.cuh"
#include "kernels_variants.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <fstream>

struct BenchmarkResult {
    std::string name;
    float avg_forward_ms;
    float avg_backward_ms;
    float total_time_s;
};

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

// MSE Loss Layer
class MSELoss : public Layer {
public:
    Matrix output;
    Matrix d_input;

    MSELoss(KernelMode m) : Layer(m) {}
    ~MSELoss() { 
        output.free(); 
        d_input.free(); 
    }

    Matrix forward(const Matrix& input) override {
        if (!output.allocated || output.rows != input.rows || output.cols != input.cols) {
            output.allocate(input.rows, input.cols);
        }
        CHECK_CUDA(cudaMemcpy(output.data, input.data, input.rows * input.cols * sizeof(float), cudaMemcpyDeviceToDevice));
        return output;
    }

    Matrix backward(const Matrix& target, float learning_rate) override {
        if (!d_input.allocated || d_input.rows != target.rows || d_input.cols != target.cols) 
            d_input.allocate(target.rows, target.cols);
        
        DISPATCH(computeMSEGradient, output, target, d_input);
        return d_input;
    }
};

BenchmarkResult run_benchmark(KernelMode mode, std::string name, Matrix& X, Matrix& Y, int epochs) {
    std::cout << "\n========================================\n";
    std::cout << "Running Benchmark: " << name << "\n";
    std::cout << "========================================\n";

    srand(1337); 

    MLP model;
    
    // Layer 1: Input (2) -> Hidden (8) with ReLU
    LinearReLU* fc1 = new LinearReLU(2, 8, mode);
    init_xavier(fc1->W); fc1->b.zeros();
    model.add(fc1);

    // Layer 2: Hidden (8) -> Output (1) (Linear)
    Linear* fc2 = new Linear(8, 1, mode);
    init_xavier(fc2->W); fc2->b.zeros();
    model.add(fc2);

    // Loss Layer
    model.add(new MSELoss(mode));

    float learning_rate = 0.1f;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        model.forward(X);
        model.backward(Y, learning_rate);
    }
    cudaDeviceSynchronize();

    auto start_time = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_forward_ms = 0.0f;
    float total_backward_ms = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        CHECK_CUDA(cudaEventRecord(start));
        Matrix preds = model.forward(X);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        total_forward_ms += milliseconds;

        // Backward
        CHECK_CUDA(cudaEventRecord(start));
        model.backward(Y, learning_rate);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        total_backward_ms += milliseconds;
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    float avg_fwd = total_forward_ms / epochs;
    float avg_bwd = total_backward_ms / epochs;
    float total_time = elapsed.count();

    std::cout << "Total Time: " << total_time << " s\n";
    std::cout << "Average Forward: " << avg_fwd << " ms\n";
    std::cout << "Average Backward: " << avg_bwd << " ms\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

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
    // Define XOR Data
    std::vector<float> h_X = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    std::vector<float> h_Y = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    Matrix X, Y;
    X.allocate(4, 2);
    Y.allocate(4, 1);
    X.copyFromHost(h_X);
    Y.copyFromHost(h_Y);

    int epochs = 10000;

    std::vector<BenchmarkResult> results;
    results.push_back(run_benchmark(KernelMode::NAIVE, "Naive", X, Y, epochs));
    results.push_back(run_benchmark(KernelMode::SHARED, "Shared Memory", X, Y, epochs));
    results.push_back(run_benchmark(KernelMode::FUSED, "Fused Kernels", X, Y, epochs));
    results.push_back(run_benchmark(KernelMode::WARP, "Warp Reduce", X, Y, epochs));

    write_results_to_json(results, "benchmark_results_xor.json");

    X.free(); Y.free();
    return 0;
}
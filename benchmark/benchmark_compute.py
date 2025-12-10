"""
Compare PyTorch vs CUDA MLP for varying number of layers
"""
import torch
import torch.nn as nn
import time
import numpy as np
import subprocess
import os
import sys
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. PyTorch Dynamic MLP
# ==========================================
class DynamicMLP(nn.Module):
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of integers [input, hidden1, hidden2, ..., output]
        """
        super(DynamicMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # ReLU after all except last layer
                self.layers.append(nn.ReLU())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)

# ==========================================
# 2. CUDA C++ MLP Generator
# ==========================================
def generate_cuda_mlp(layer_sizes, output_path="src/main_benchmark_compute.cu"):
    """Generate CUDA code for dynamic MLP"""
    
    num_layers = len(layer_sizes) - 1
    
    code = f"""#include "utils.cuh"
#include "kernels.cuh"
#include "mlp.cuh"
#include <iostream>
#include <vector>
#include <chrono>

void init_xavier(Matrix& m) {{
    float scale = sqrt(2.0f / m.rows);
    std::vector<float> host_data(m.rows * m.cols);
    for (size_t i = 0; i < host_data.size(); ++i) {{
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        host_data[i] = (r * 2.0f - 1.0f) * scale;
    }}
    m.copyFromHost(host_data);
}}

int main() {{
    srand(1337);
    
    // Synthetic data
    int n_samples = 60000;
    int batch_size = 64;
    int epochs = 5;
    float learning_rate = 0.01f;
    
    Matrix full_X, full_Y;
    full_X.allocate(n_samples, {layer_sizes[0]});
    full_Y.allocate(n_samples, {layer_sizes[-1]});
    
    // Random initialization
    std::vector<float> h_X(n_samples * {layer_sizes[0]});
    std::vector<float> h_Y(n_samples * {layer_sizes[-1]});
    for (auto& v : h_X) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (auto& v : h_Y) v = ((float)rand() / RAND_MAX);
    
    full_X.copyFromHost(h_X);
    full_Y.copyFromHost(h_Y);
    
    // Build Model
    MLP model;
"""
    
    # Generate layers
    for i in range(num_layers):
        code += f"""    
    Linear* fc{i+1} = new Linear({layer_sizes[i]}, {layer_sizes[i+1]});
    init_xavier(fc{i+1}->W);
    fc{i+1}->b.zeros();
    model.add(fc{i+1});
"""
        if i < num_layers - 1:  # ReLU after all except last
            code += f"    model.add(new ReLU());\n"
    
    code += f"""
    // Allocate batches
    int num_batches = n_samples / batch_size;
    Matrix batch_X, batch_Y, d_loss;
    batch_X.allocate(batch_size, {layer_sizes[0]});
    batch_Y.allocate(batch_size, {layer_sizes[-1]});
    d_loss.allocate(batch_size, {layer_sizes[-1]});
    
    // Warmup
    for (int b = 0; b < 10 && b < num_batches; ++b) {{
        CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * {layer_sizes[0]},
                              batch_size * {layer_sizes[0]} * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * {layer_sizes[-1]},
                              batch_size * {layer_sizes[-1]} * sizeof(float), cudaMemcpyDeviceToDevice));
        Matrix preds = model.forward(batch_X);
        computeMSEGradient(preds, batch_Y, d_loss);
        model.backward(d_loss, learning_rate);
    }}
    
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {{
        for (int b = 0; b < num_batches; ++b) {{
            CHECK_CUDA(cudaMemcpy(batch_X.data, full_X.data + b * batch_size * {layer_sizes[0]},
                                  batch_size * {layer_sizes[0]} * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(batch_Y.data, full_Y.data + b * batch_size * {layer_sizes[-1]},
                                  batch_size * {layer_sizes[-1]} * sizeof(float), cudaMemcpyDeviceToDevice));
            
            Matrix preds = model.forward(batch_X);
            computeMSEGradient(preds, batch_Y, d_loss);
            model.backward(d_loss, learning_rate);
        }}
    }}
    
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Training Time: " << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}}
"""
    
    with open(output_path, 'w+') as f:
        f.write(code)
    
    return output_path

# ==========================================
# 3. Benchmark Configuration
# ==========================================
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
N_SAMPLES = 60000

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
N_SAMPLES = 60000

def benchmark_pytorch(layer_sizes):
    """Benchmark PyTorch implementation"""
    if not torch.cuda.is_available():
        print("CUDA not available for PyTorch")
        return None
    
    device = torch.device('cuda')
    
    # Generate synthetic data
    full_X = torch.randn(N_SAMPLES, layer_sizes[0], device=device)
    full_Y = torch.randn(N_SAMPLES, layer_sizes[-1], device=device)
    
    # Setup model
    model = DynamicMLP(layer_sizes).to(device)
    model.init_xavier()
    
    criterion = nn.MSELoss()
    
    # Warmup
    for b in range(10):
        start_idx = b * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_X = full_X[start_idx:end_idx]
        batch_Y = full_Y[start_idx:end_idx]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= LR * param.grad
    
    torch.cuda.synchronize()
    
    # Benchmark
    num_batches = N_SAMPLES // BATCH_SIZE
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        for b in range(num_batches):
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_X = full_X[start_idx:end_idx]
            batch_Y = full_Y[start_idx:end_idx]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    param -= LR * param.grad
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    return elapsed

def benchmark_cuda(layer_sizes):
    """Benchmark CUDA implementation"""
    
    # Generate CUDA code
    cuda_file = generate_cuda_mlp(layer_sizes)
    executable = "../bin/mlp_benchmark_compute"
    
    # Compile
    print(f"  Compiling CUDA code...")
    compile_cmd = [
        "nvcc",
        "-arch=sm_75",
        "-O3",
        "-std=c++17",
        "-o", executable,
        cuda_file,
        "src/kernels.cu"
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Compilation error: {e}")
        return None
    
    # Run
    print(f"  Running CUDA benchmark...")
    try:
        result = subprocess.run([executable], capture_output=True, text=True, timeout=120)
        
        # Parse time from output
        for line in result.stdout.split('\n'):
            if "Training Time:" in line:
                time_str = line.split(':')[1].strip().split()[0]
                return float(time_str)
        
        print("Could not parse training time from CUDA output")
        return None
        
    except Exception as e:
        print(f"CUDA execution error: {e}")
        return None

def run_benchmark():
    print("=" * 70)
    print("PyTorch vs CUDA MLP: Varying Number of Layers")
    print("=" * 70)
    print(f"Configuration: {N_SAMPLES} samples, {BATCH_SIZE} batch size, {EPOCHS} epochs")
    print()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test configurations: varying number of layers
    # Format: [input, hidden1, hidden2, ..., output]
    configs = [
        ([784, 10], "1 hidden layer (784->10)"),
        ([784, 256, 10], "2 layers (784->256->10)"),
        ([784, 256, 128, 10], "3 layers (784->256->128->10)"),
        ([784, 256, 128, 64, 10], "4 layers"),
        ([784, 256, 128, 64, 32, 10], "5 layers"),
        ([784, 256, 128, 64, 32, 16, 10], "6 layers"),
        ([784, 256, 128, 64, 32, 16, 16, 10], "7 layers"),
        ([784, 256, 128, 64, 32, 16, 16, 16, 10], "8 layers"),
        ([784, 256, 128, 64, 32, 16, 16, 16, 16, 10], "9 layers"),
        ([784, 256, 128, 64, 32, 16, 16, 16, 16, 16, 10], "10 layers"),
    ]
    
    results = []
    
    for layer_sizes, description in configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {description}")
        print(f"Architecture: {' -> '.join(map(str, layer_sizes))}")
        print(f"{'=' * 70}")
        
        # PyTorch
        print("Running PyTorch...")
        pytorch_time = benchmark_pytorch(layer_sizes)
        
        # CUDA
        print("Running CUDA...")
        cuda_time = benchmark_cuda(layer_sizes)
        
        if pytorch_time and cuda_time:
            speedup = pytorch_time / cuda_time
            results.append({
                'layers': len(layer_sizes) - 1,
                'architecture': ' -> '.join(map(str, layer_sizes)),
                'pytorch_time': pytorch_time,
                'cuda_time': cuda_time,
                'speedup': speedup
            })
            
            print(f"\nResults:")
            print(f"  PyTorch: {pytorch_time:.4f}s")
            print(f"  CUDA:    {cuda_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x {'(CUDA faster)' if speedup > 1 else '(PyTorch faster)'}")
        else:
            print("  Benchmark failed!")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Layers':<10} {'Architecture':<35} {'PyTorch':<12} {'CUDA':<12} {'Speedup':<10}")
    print(f"{'-' * 70}")
    
    for r in results:
        print(f"{r['layers']:<10} {r['architecture']:<35} {r['pytorch_time']:<12.4f} {r['cuda_time']:<12.4f} {r['speedup']:<10.2f}x")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/layer_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/layer_comparison.json")

if __name__ == "__main__":
    run_benchmark()
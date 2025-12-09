"""
Benchmark XOR training: CUDA vs PyTorch
"""
import os
import sys
import json
import time
import subprocess
import torch
from pytorch_mlp import XORMLP, train_xor

# Configuration
CUDA_EXECUTABLE = "../bin/mlp_xor_test"
RESULTS_DIR = "./results"
EPOCHS = 10000
LEARNING_RATE = 0.01


def benchmark_cuda_xor():
    """Run CUDA XOR implementation and measure performance"""
    print("=" * 60)
    print("Running CUDA MLP on XOR...")
    print("=" * 60)
    
    if not os.path.exists(CUDA_EXECUTABLE):
        print(f"Error: CUDA executable not found at {CUDA_EXECUTABLE}")
        print("Please run 'make xor' first to build the CUDA XOR implementation.")
        return None
    
    start_time = time.time()
    
    # Run CUDA executable and capture output
    try:
        result = subprocess.run(
            [CUDA_EXECUTABLE],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        # Parse output for loss values and training time
        output_lines = result.stdout.split('\n')
        epochs = []
        losses = []
        training_time = None
        
        for line in output_lines:
            if "Epoch" in line and "Loss:" in line:
                # Extract from line like "Epoch  5000 | Loss: 0.123456"
                parts = line.split('|')
                if len(parts) >= 2:
                    epoch = int(parts[0].split()[1])
                    loss = float(parts[1].split(':')[1].strip())
                    epochs.append(epoch)
                    losses.append(loss)
            elif "Training Time:" in line:
                # Extract from line like "Training Time: 1.234 seconds"
                training_time = float(line.split(':')[1].split()[0])
        
        # Fallback to subprocess time if training time not found
        if training_time is None:
            training_time = time.time() - start_time
            print("Warning: Could not parse training time from output, using subprocess time")
        
        total_time = training_time
        
        results = {
            'implementation': 'CUDA',
            'total_time': total_time,
            'epochs': epochs,
            'losses': losses,
            'final_loss': losses[-1] if losses else None,
            'learning_rate': LEARNING_RATE,
            'total_epochs': EPOCHS,
            'output': result.stdout
        }
        
        print(f"\nCUDA Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Final Loss: {losses[-1] if losses else 'N/A'}")
        
        return results
        
    except subprocess.TimeoutExpired:
        print("Error: CUDA execution timed out")
        return None
    except Exception as e:
        print(f"Error running CUDA implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_pytorch_xor(device='cuda'):
    """Run PyTorch XOR implementation and measure performance"""
    print("\n" + "=" * 60)
    print(f"Running PyTorch MLP on XOR (device: {device})...")
    print("=" * 60)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model = XORMLP()
    
    try:
        history, total_time = train_xor(
            model=model,
            device=device,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        
        results = {
            'implementation': f'PyTorch ({device})',
            'device': device,
            'total_time': total_time,
            'epochs': history['epoch'],
            'losses': history['loss'],
            'final_loss': history['loss'][-1],
            'learning_rate': LEARNING_RATE,
            'total_epochs': EPOCHS
        }
        
        print(f"\nPyTorch ({device}) Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Final Loss: {history['loss'][-1]:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error running PyTorch implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results_list, filename='xor_benchmark.json'):
    """Save benchmark results to JSON file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def compare_results(results_list):
    """Print comparison summary"""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Implementation':<20} {'Time (s)':<12} {'Final Loss':<15} {'Speedup':<10}")
    print("-" * 60)
    
    # Find baseline (CUDA) time
    cuda_time = None
    for result in results_list:
        if result and 'CUDA' in result['implementation']:
            cuda_time = result['total_time']
            break
    
    for result in results_list:
        if result is None:
            continue
            
        impl = result['implementation']
        total_time = result['total_time']
        final_loss = result['final_loss']
        
        speedup = ""
        if cuda_time and total_time > 0:
            speedup = f"{cuda_time / total_time:.2f}x"
        
        print(f"{impl:<20} {total_time:<12.3f} {final_loss:<15.6f} {speedup:<10}")
    
    print("\n" + "=" * 60)


def main():
    print("XOR Benchmark: CUDA vs PyTorch")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print()
    
    results = []
    
    # Benchmark CUDA
    cuda_results = benchmark_cuda_xor()
    if cuda_results:
        results.append(cuda_results)
    
    # Benchmark PyTorch GPU
    pytorch_gpu_results = benchmark_pytorch_xor(device='cuda')
    if pytorch_gpu_results:
        results.append(pytorch_gpu_results)
    
    # Benchmark PyTorch CPU (optional)
    # pytorch_cpu_results = benchmark_pytorch_xor(device='cpu')
    # if pytorch_cpu_results:
    #     results.append(pytorch_cpu_results)
    
    # Save and compare
    if results:
        save_results(results)
        compare_results(results)
    else:
        print("No successful benchmark runs to compare")


if __name__ == '__main__':
    main()

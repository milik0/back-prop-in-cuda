"""
Benchmark MNIST training: CUDA vs PyTorch
Fixed timing methodology for fair comparison
"""
import os
import sys
import json
import time
import subprocess
import torch
from pytorch_mlp import MLP, train_mnist

# Configuration
DATA_PATH = os.path.expanduser("~/data")  # Adjust if needed
CUDA_EXECUTABLE = "../bin/mlp_test"
RESULTS_DIR = "./results"
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def benchmark_cuda_mnist():
    """Run CUDA implementation and measure performance"""
    print("=" * 60)
    print("Running CUDA MLP on MNIST...")
    print("=" * 60)
    
    if not os.path.exists(CUDA_EXECUTABLE):
        print(f"Error: CUDA executable not found at {CUDA_EXECUTABLE}")
        print("Please run 'make' first to build the CUDA implementation.")
        return None
    
    # Measure total wall-clock time (including process startup)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [CUDA_EXECUTABLE],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Total wall-clock time
        total_time = time.time() - start_time
        
        # Parse output for accuracy and internal timing info
        output_lines = result.stdout.split('\n')
        train_accuracies = []
        test_accuracy = None
        kernel_time = None
        
        for line in output_lines:
            if "Avg Accuracy:" in line:
                # Extract accuracy from line like "Epoch 0 | Avg Accuracy: 85.5%"
                try:
                    acc = float(line.split("Avg Accuracy: ")[1].split("%")[0])
                    train_accuracies.append(acc)
                except (ValueError, IndexError):
                    continue
            elif "Test Set Accuracy:" in line:
                try:
                    test_accuracy = float(line.split("Test Set Accuracy: ")[1].split("%")[0])
                except (ValueError, IndexError):
                    pass
            elif "Training Time:" in line:
                # Extract kernel/training time for reference
                try:
                    kernel_time = float(line.split(':')[1].split()[0])
                except (ValueError, IndexError):
                    pass
        
        results = {
            'implementation': 'CUDA',
            'total_time': total_time,
            'kernel_time': kernel_time,  # Internal training time
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'output': result.stdout
        }
        
        print(f"\nCUDA Results:")
        print(f"  Total Wall Time: {total_time:.3f}s")
        if kernel_time:
            print(f"  Kernel Time: {kernel_time:.3f}s")
        print(f"  Final Train Accuracy: {train_accuracies[-1] if train_accuracies else 'N/A'}%")
        print(f"  Test Accuracy: {test_accuracy}%")
        
        return results
        
    except subprocess.TimeoutExpired:
        print("Error: CUDA execution timed out")
        return None
    except Exception as e:
        print(f"Error running CUDA implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_pytorch_mnist(device='cuda', warmup=True):
    """Run PyTorch implementation and measure performance"""
    print("\n" + "=" * 60)
    print(f"Running PyTorch MLP on MNIST (device: {device})...")
    print("=" * 60)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Warmup GPU if needed
    if warmup and device == 'cuda':
        print("Warming up GPU...")
        dummy_model = MLP()
        try:
            train_mnist(
                dummy_model, 
                device=device, 
                data_path=DATA_PATH,
                epochs=1, 
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
            )
            torch.cuda.synchronize()
            del dummy_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    model = MLP()
    
    try:
        # Measure total time including all overhead
        start_time = time.time()
        
        history, training_time = train_mnist(
            model=model,
            device=device,
            data_path=DATA_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        
        # Ensure GPU operations complete
        if device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        
        results = {
            'implementation': f'PyTorch ({device})',
            'device': device,
            'total_time': total_time,
            'training_time': training_time,  # Time from train_mnist function
            'train_accuracies': history['train_acc'],
            'test_accuracy': history['test_acc'][-1],
            'epoch_times': history['time'],
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
        
        print(f"\nPyTorch ({device}) Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Training Time: {training_time:.3f}s")
        print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  Test Accuracy: {history['test_acc'][-1]:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error running PyTorch implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results_list, filename='mnist_benchmark.json'):
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
    
    print("\nComparison using TOTAL WALL-CLOCK TIME (most fair):")
    print(f"{'Implementation':<20} {'Total Time':<12} {'Train Acc':<12} {'Test Acc':<12} {'Speedup':<10}")
    print("-" * 75)
    
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
        train_acc = result['train_accuracies'][-1] if result['train_accuracies'] else 0
        test_acc = result['test_accuracy']
        
        speedup = ""
        if cuda_time and total_time > 0:
            speedup = f"{cuda_time / total_time:.2f}x"
        
        print(f"{impl:<20} {total_time:<12.3f} {train_acc:<12.2f} {test_acc:<12.2f} {speedup:<10}")
    
    # Also show kernel/training time breakdown
    print("\n\nInternal Training Time Breakdown:")
    print(f"{'Implementation':<20} {'Kernel/Train Time':<18} {'Overhead':<12}")
    print("-" * 75)
    
    for result in results_list:
        if result is None:
            continue
            
        impl = result['implementation']
        total_time = result['total_time']
        
        if 'CUDA' in result['implementation'] and result.get('kernel_time'):
            internal = result['kernel_time']
            overhead = total_time - internal
            print(f"{impl:<20} {internal:<18.3f} {overhead:<12.3f}")
        elif 'PyTorch' in result['implementation'] and result.get('training_time'):
            internal = result['training_time']
            overhead = total_time - internal
            print(f"{impl:<20} {internal:<18.3f} {overhead:<12.3f}")
    
    print("\n" + "=" * 60)
    print("NOTE: Speedup is calculated using total wall-clock time for fair comparison")
    print("=" * 60)


def main():
    print("MNIST Benchmark: CUDA vs PyTorch")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Data Path: {DATA_PATH}")
    print()
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at {DATA_PATH}")
        print("Please download MNIST data first (see README.md)")
        return
    
    results = []
    
    # Benchmark CUDA
    cuda_results = benchmark_cuda_mnist()
    if cuda_results:
        results.append(cuda_results)
    
    # Benchmark PyTorch GPU
    pytorch_gpu_results = benchmark_pytorch_mnist(device='cuda')
    if pytorch_gpu_results:
        results.append(pytorch_gpu_results)
    
    # Benchmark PyTorch CPU (optional)
    # pytorch_cpu_results = benchmark_pytorch_mnist(device='cpu')
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
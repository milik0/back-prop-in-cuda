"""
Benchmark XOR training: CUDA vs PyTorch
Fixed timing methodology for fair comparison
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
    
    # Measure total wall-clock time (including process startup)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [CUDA_EXECUTABLE],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Total wall-clock time
        total_time = time.time() - start_time
        
        # Parse output for loss values and internal training time
        output_lines = result.stdout.split('\n')
        epochs = []
        losses = []
        kernel_time = None
        
        for line in output_lines:
            if "Epoch" in line and "Loss:" in line:
                # Extract from line like "Epoch  5000 | Loss: 0.123456"
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        epoch = int(parts[0].split()[1])
                        loss = float(parts[1].split(':')[1].strip())
                        epochs.append(epoch)
                        losses.append(loss)
                    except (ValueError, IndexError):
                        continue
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
            'epochs': epochs,
            'losses': losses,
            'final_loss': losses[-1] if losses else None,
            'learning_rate': LEARNING_RATE,
            'total_epochs': EPOCHS,
            'output': result.stdout
        }
        
        print(f"\nCUDA Results:")
        print(f"  Total Wall Time: {total_time:.3f}s")
        if kernel_time:
            print(f"  Kernel Time: {kernel_time:.3f}s")
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


def benchmark_pytorch_xor(device='cuda', warmup=True):
    """Run PyTorch XOR implementation and measure performance"""
    print("\n" + "=" * 60)
    print(f"Running PyTorch MLP on XOR (device: {device})...")
    print("=" * 60)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Warmup GPU if needed
    if warmup and device == 'cuda':
        print("Warming up GPU...")
        dummy_model = XORMLP()
        train_xor(dummy_model, device=device, epochs=100, lr=LEARNING_RATE)
        torch.cuda.synchronize()
        del dummy_model
        torch.cuda.empty_cache()
    
    model = XORMLP()
    
    try:
        # Measure total time including all overhead
        start_time = time.time()
        
        history, training_time = train_xor(
            model=model,
            device=device,
            epochs=EPOCHS,
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
            'training_time': training_time,  # Time from train_xor function
            'epochs': history['epoch'],
            'losses': history['loss'],
            'final_loss': history['loss'][-1],
            'learning_rate': LEARNING_RATE,
            'total_epochs': EPOCHS
        }
        
        print(f"\nPyTorch ({device}) Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Training Time: {training_time:.3f}s")
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
    
    print("\nComparison using TOTAL WALL-CLOCK TIME (most fair):")
    print(f"{'Implementation':<20} {'Total Time':<12} {'Final Loss':<15} {'Speedup':<10}")
    print("-" * 70)
    
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
    
    # Also show kernel/training time breakdown
    print("\n\nInternal Training Time Breakdown:")
    print(f"{'Implementation':<20} {'Kernel/Train Time':<18} {'Overhead':<12}")
    print("-" * 70)
    
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
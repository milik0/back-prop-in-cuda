"""
Benchmark Fashion-MNIST training: CUDA vs PyTorch
"""
import os
import sys
import json
import time
import subprocess
import torch
from pytorch_mlp import FashionMLP, train_fashion_mnist

# Configuration
CUDA_EXECUTABLE = "../bin/mlp_fashion_mnist_test"
RESULTS_DIR = "./results"
DATA_PATH = os.path.expanduser("~/data/fashion_mnist")
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def benchmark_cuda_fashion_mnist():
    """Run CUDA implementation and measure performance"""
    print("=" * 60)
    print("Running CUDA MLP on Fashion-MNIST...")
    print("=" * 60)
    
    if not os.path.exists(CUDA_EXECUTABLE):
        print(f"Error: CUDA executable not found at {CUDA_EXECUTABLE}")
        print("Please run 'make fashion-mnist' first to build the CUDA implementation.")
        return None
    
    start_time = time.time()
    
    # Prepare data paths
    train_images = f"{DATA_PATH}/train-images-idx3-ubyte"
    train_labels = f"{DATA_PATH}/train-labels-idx1-ubyte"
    test_images = f"{DATA_PATH}/t10k-images-idx3-ubyte"
    test_labels = f"{DATA_PATH}/t10k-labels-idx1-ubyte"
    
    # Run CUDA executable and capture output
    try:
        result = subprocess.run(
            [CUDA_EXECUTABLE, train_images, train_labels, test_images, test_labels],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse output for accuracy and timing info
        output_lines = result.stdout.split('\n')
        train_accuracies = []
        test_accuracy = None
        training_time = None
        
        for line in output_lines:
            if "Avg Accuracy:" in line:
                acc = float(line.split("Avg Accuracy: ")[1].split("%")[0])
                train_accuracies.append(acc)
            elif "Test Set Accuracy:" in line:
                test_accuracy = float(line.split("Test Set Accuracy: ")[1].split("%")[0])
            elif "Training Time:" in line:
                training_time = float(line.split(':')[1].split()[0])
        
        # Fallback to subprocess time if training time not found
        if training_time is None:
            training_time = time.time() - start_time
            print("Warning: Could not parse training time from output, using subprocess time")
        
        total_time = training_time
        
        results = {
            'implementation': 'CUDA',
            'total_time': total_time,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'output': result.stdout
        }
        
        print(f"\nCUDA Results:")
        print(f"  Total Time: {total_time:.3f}s")
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


def benchmark_pytorch_fashion_mnist(device='cuda'):
    """Run PyTorch implementation and measure performance"""
    print("\n" + "=" * 60)
    print(f"Running PyTorch MLP on Fashion-MNIST (device: {device})...")
    print("=" * 60)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model = FashionMLP()
    
    try:
        history, total_time = train_fashion_mnist(
            model=model,
            device=device,
            data_path=DATA_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        
        results = {
            'implementation': f'PyTorch ({device})',
            'device': device,
            'total_time': total_time,
            'train_accuracies': history['train_acc'],
            'test_accuracy': history['test_acc'][-1],
            'epoch_times': history['time'],
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
        
        print(f"\nPyTorch ({device}) Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  Test Accuracy: {history['test_acc'][-1]:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error running PyTorch implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results_list, filename='fashion_mnist_benchmark.json'):
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
    
    print(f"\n{'Implementation':<20} {'Time (s)':<12} {'Train Acc':<12} {'Test Acc':<12} {'Speedup':<10}")
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
        train_acc = result['train_accuracies'][-1] if result['train_accuracies'] else 0
        test_acc = result['test_accuracy']
        
        speedup = ""
        if cuda_time and total_time > 0:
            speedup = f"{cuda_time / total_time:.2f}x"
        
        print(f"{impl:<20} {total_time:<12.3f} {train_acc:<12.2f} {test_acc:<12.2f} {speedup:<10}")
    
    print("\n" + "=" * 60)


def main():
    print("Fashion-MNIST Benchmark: CUDA vs PyTorch")
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
        print("Please download Fashion-MNIST data first")
        return
    
    results = []
    
    # Benchmark CUDA
    cuda_results = benchmark_cuda_fashion_mnist()
    if cuda_results:
        results.append(cuda_results)
    
    # Benchmark PyTorch GPU
    pytorch_gpu_results = benchmark_pytorch_fashion_mnist(device='cuda')
    if pytorch_gpu_results:
        results.append(pytorch_gpu_results)
    
    # Save and compare
    if results:
        save_results(results)
        compare_results(results)
    else:
        print("No successful benchmark runs to compare")


if __name__ == '__main__':
    main()






















































































































    main()if __name__ == '__main__':        print("No successful benchmark runs to compare")    else:        compare_results(results)        save_results(results)    if results:    # Save and compare            results.append(pytorch_gpu_results)    if pytorch_gpu_results:    pytorch_gpu_results = benchmark_pytorch_breast_cancer(device='cuda')    # Benchmark PyTorch GPU            results.append(cuda_results)    if cuda_results:    cuda_results = benchmark_cuda_breast_cancer()    # Benchmark CUDA        results = []            return        print("Download from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")        print("Please download Breast Cancer Wisconsin dataset first")        print(f"Error: CSV file not found at {CSV_PATH}")    if not os.path.exists(CSV_PATH):    # Check if data exists        print()    print(f"  CSV Path: {CSV_PATH}")    print(f"  Learning Rate: {LEARNING_RATE}")    print(f"  Batch Size: {BATCH_SIZE}")    print(f"  Epochs: {EPOCHS}")    print(f"Configuration:")    print("=" * 60)    print("Breast Cancer Wisconsin Benchmark: CUDA vs PyTorch")def main():    print("\n" + "=" * 60)            print(f"{impl:<20} {total_time:<12.3f} {train_acc:<12.2f} {test_acc:<12.2f} {speedup:<10}")                    speedup = f"{cuda_time / total_time:.2f}x"        if cuda_time and total_time > 0:        speedup = ""                test_acc = result['test_accuracy']        train_acc = result['train_accuracies'][-1] if result['train_accuracies'] else 0        total_time = result['total_time']        impl = result['implementation']                        continue        if result is None:    for result in results_list:                break            cuda_time = result['total_time']        if result and 'CUDA' in result['implementation']:    for result in results_list:    cuda_time = None    # Find baseline (CUDA) time        print("-" * 70)    print(f"\n{'Implementation':<20} {'Time (s)':<12} {'Train Acc':<12} {'Test Acc':<12} {'Speedup':<10}")        print("=" * 60)    print("BENCHMARK COMPARISON SUMMARY")    print("\n" + "=" * 60)    """Print comparison summary"""def compare_results(results_list):    print(f"\nResults saved to {filepath}")            json.dump(results_list, f, indent=2)    with open(filepath, 'w') as f:        filepath = os.path.join(RESULTS_DIR, filename)    os.makedirs(RESULTS_DIR, exist_ok=True)    """Save benchmark results to JSON file"""def save_results(results_list, filename='breast_cancer_benchmark.json'):        return None        traceback.print_exc()        import traceback        print(f"Error running PyTorch implementation: {e}")    except Exception as e:                return results                print(f"  Test Accuracy: {history['test_acc'][-1]:.2f}%")        print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")        print(f"  Total Time: {total_time:.3f}s")        print(f"\nPyTorch ({device}) Results:")                }            'learning_rate': LEARNING_RATE            'batch_size': BATCH_SIZE,            'epochs': EPOCHS,            'epoch_times': history['time'],            'test_accuracy': history['test_acc'][-1],            'train_accuracies': history['train_acc'],            'total_time': total_time,            'device': device,            'implementation': f'PyTorch ({device})',        results = {                )            lr=LEARNING_RATE            batch_size=BATCH_SIZE,            epochs=EPOCHS,            csv_path=CSV_PATH,            device=device,            model=model,        history, total_time = train_breast_cancer(    try:        model = BreastCancerMLP()            device = 'cpu'        print("Warning: CUDA not available, falling back to CPU")    if device == 'cuda' and not torch.cuda.is_available():        print("=" * 60)    print(f"Running PyTorch MLP on Breast Cancer Wisconsin (device: {device})...")    print("\n" + "=" * 60)    """Run PyTorch implementation and measure performance"""def benchmark_pytorch_breast_cancer(device='cuda'):        return None        traceback.print_exc()        import traceback        print(f"Error running CUDA implementation: {e}")    except Exception as e:        return None        print("Error: CUDA execution timed out")    except subprocess.TimeoutExpired:                return results                print(f"  Test Accuracy: {test_accuracy}%")        print(f"  Final Train Accuracy: {train_accuracies[-1] if train_accuracies else 'N/A'}%")        print(f"  Total Time: {total_time:.3f}s")        print(f"\nCUDA Results:")                }            'output': result.stdout            'learning_rate': LEARNING_RATE,            'batch_size': BATCH_SIZE,            'epochs': EPOCHS,            'test_accuracy': test_accuracy,            'train_accuracies': train_accuracies,            'total_time': total_time,            'implementation': 'CUDA',        results = {                total_time = training_time                    print("Warning: Could not parse training time from output, using subprocess time")            training_time = time.time() - start_time        if training_time is None:        # Fallback to subprocess time if training time not found                        training_time = float(line.split(':')[1].split()[0])            elif "Training Time:" in line:                test_accuracy = float(line.split("Test Set Accuracy: ")[1].split("%")[0])            elif "Test Set Accuracy:" in line:                train_accuracies.append(acc)                acc = float(line.split("Avg Accuracy: ")[1].split("%")[0])            if "Avg Accuracy:" in line and "Epoch" in line:        for line in output_lines:                training_time = None        test_accuracy = None        train_accuracies = []        output_lines = result.stdout.split('\n')        # Parse output for accuracy and timing info                )            timeout=300  # 5 minute timeout            text=True,            capture_output=True,            [CUDA_EXECUTABLE, CSV_PATH],        result = subprocess.run(    try:    # Run CUDA executable and capture output        start_time = time.time()            return None        print("Please run 'make breast-cancer' first to build the CUDA implementation.")        print(f"Error: CUDA executable not found at {CUDA_EXECUTABLE}")    if not os.path.exists(CUDA_EXECUTABLE):        print("=" * 60)    print("Running CUDA MLP on Breast Cancer Wisconsin...")    print("=" * 60)    """Run CUDA implementation and measure performance"""def benchmark_cuda_breast_cancer():LEARNING_RATE = 0.01BATCH_SIZE = 32EPOCHS = 50CSV_PATH = os.path.expanduser("~/data/breast_cancer.csv")RESULTS_DIR = "./results"CUDA_EXECUTABLE = "../bin/mlp_breast_cancer_test"# Configurationfrom pytorch_mlp import BreastCancerMLP, train_breast_cancerimport torchimport subprocessimport timeimport jsonimport sysimport os"""Benchmark Breast Cancer Wisconsin training: CUDA vs PyTorchBenchmark Fashion-MNIST training: CUDA vs PyTorch
"""
import os
import sys
import json
import time
import subprocess
import torch
from pytorch_mlp import FashionMLP, train_fashion_mnist

# Configuration
CUDA_EXECUTABLE = "../bin/mlp_fashion_mnist_test"
RESULTS_DIR = "./results"
DATA_PATH = os.path.expanduser("~/data/fashion_mnist")
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01


def benchmark_cuda_fashion_mnist():
    """Run CUDA implementation and measure performance"""
    print("=" * 60)
    print("Running CUDA MLP on Fashion-MNIST...")
    print("=" * 60)
    
    if not os.path.exists(CUDA_EXECUTABLE):
        print(f"Error: CUDA executable not found at {CUDA_EXECUTABLE}")
        print("Please run 'make fashion-mnist' first to build the CUDA implementation.")
        return None
    
    start_time = time.time()
    
    # Prepare data paths
    train_images = f"{DATA_PATH}/train-images-idx3-ubyte"
    train_labels = f"{DATA_PATH}/train-labels-idx1-ubyte"
    test_images = f"{DATA_PATH}/t10k-images-idx3-ubyte"
    test_labels = f"{DATA_PATH}/t10k-labels-idx1-ubyte"
    
    # Run CUDA executable and capture output
    try:
        result = subprocess.run(
            [CUDA_EXECUTABLE, train_images, train_labels, test_images, test_labels],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse output for accuracy and timing info
        output_lines = result.stdout.split('\n')
        train_accuracies = []
        test_accuracy = None
        training_time = None
        
        for line in output_lines:
            if "Avg Accuracy:" in line:
                acc = float(line.split("Avg Accuracy: ")[1].split("%")[0])
                train_accuracies.append(acc)
            elif "Test Set Accuracy:" in line:
                test_accuracy = float(line.split("Test Set Accuracy: ")[1].split("%")[0])
            elif "Training Time:" in line:
                training_time = float(line.split(':')[1].split()[0])
        
        # Fallback to subprocess time if training time not found
        if training_time is None:
            training_time = time.time() - start_time
            print("Warning: Could not parse training time from output, using subprocess time")
        
        total_time = training_time
        
        results = {
            'implementation': 'CUDA',
            'total_time': total_time,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'output': result.stdout
        }
        
        print(f"\nCUDA Results:")
        print(f"  Total Time: {total_time:.3f}s")
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


def benchmark_pytorch_fashion_mnist(device='cuda'):
    """Run PyTorch implementation and measure performance"""
    print("\n" + "=" * 60)
    print(f"Running PyTorch MLP on Fashion-MNIST (device: {device})...")
    print("=" * 60)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model = FashionMLP()
    
    try:
        history, total_time = train_fashion_mnist(
            model=model,
            device=device,
            data_path=DATA_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        
        results = {
            'implementation': f'PyTorch ({device})',
            'device': device,
            'total_time': total_time,
            'train_accuracies': history['train_acc'],
            'test_accuracy': history['test_acc'][-1],
            'epoch_times': history['time'],
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
        
        print(f"\nPyTorch ({device}) Results:")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"  Test Accuracy: {history['test_acc'][-1]:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error running PyTorch implementation: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results_list, filename='fashion_mnist_benchmark.json'):
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
    
    print(f"\n{'Implementation':<20} {'Time (s)':<12} {'Train Acc':<12} {'Test Acc':<12} {'Speedup':<10}")
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
        train_acc = result['train_accuracies'][-1] if result['train_accuracies'] else 0
        test_acc = result['test_accuracy']
        
        speedup = ""
        if cuda_time and total_time > 0:
            speedup = f"{cuda_time / total_time:.2f}x"
        
        print(f"{impl:<20} {total_time:<12.3f} {train_acc:<12.2f} {test_acc:<12.2f} {speedup:<10}")
    
    print("\n" + "=" * 60)


def main():
    print("Fashion-MNIST Benchmark: CUDA vs PyTorch")
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
        print("Please download Fashion-MNIST data first")
        return
    
    results = []
    
    # Benchmark CUDA
    cuda_results = benchmark_cuda_fashion_mnist()
    if cuda_results:
        results.append(cuda_results)
    
    # Benchmark PyTorch GPU
    pytorch_gpu_results = benchmark_pytorch_fashion_mnist(device='cuda')
    if pytorch_gpu_results:
        results.append(pytorch_gpu_results)
    
    # Save and compare
    if results:
        save_results(results)
        compare_results(results)
    else:
        print("No successful benchmark runs to compare")


if __name__ == '__main__':
    main()

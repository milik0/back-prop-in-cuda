"""
Visualize benchmark results comparing CUDA and PyTorch implementations
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./results"


def plot_mnist_results(filename='mnist_benchmark.json'):
    """Plot MNIST benchmark results"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Results file not found at {filepath}")
        print("Please run benchmark_mnist.py first")
        return
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MNIST Benchmark: CUDA vs PyTorch', fontsize=16, fontweight='bold')
    
    # Extract data
    implementations = []
    train_accs = []
    test_accs = []
    times = []
    
    for result in results:
        implementations.append(result['implementation'])
        train_accs.append(result['train_accuracies'])
        test_accs.append(result['test_accuracy'])
        times.append(result['total_time'])
    
    # Plot 1: Training Accuracy over Epochs
    ax1 = axes[0, 0]
    for impl, accs in zip(implementations, train_accs):
        epochs = list(range(len(accs)))
        ax1.plot(epochs, accs, marker='o', label=impl, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy per Epoch', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy Comparison
    ax2 = axes[0, 1]
    x_pos = np.arange(len(implementations))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax2.bar(x_pos, test_accs, color=colors[:len(implementations)], alpha=0.7)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Final Test Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(implementations, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Total Training Time
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, times, color=colors[:len(implementations)], alpha=0.7)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Total Training Time', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(implementations, rotation=15, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Speedup Comparison
    ax4 = axes[1, 1]
    cuda_time = times[0]  # Assume first result is CUDA
    speedups = [cuda_time / t for t in times]
    bars = ax4.bar(x_pos, speedups, color=colors[:len(implementations)], alpha=0.7)
    ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline (CUDA)')
    ax4.set_ylabel('Speedup (relative to CUDA)', fontsize=12)
    ax4.set_title('Performance Speedup', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(implementations, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(RESULTS_DIR, 'mnist_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.show()


def plot_xor_results(filename='xor_benchmark.json'):
    """Plot XOR benchmark results"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Results file not found at {filepath}")
        print("Please run benchmark_xor.py first")
        return
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('XOR Benchmark: CUDA vs PyTorch', fontsize=16, fontweight='bold')
    
    # Extract data
    implementations = []
    all_epochs = []
    all_losses = []
    times = []
    final_losses = []
    
    for result in results:
        implementations.append(result['implementation'])
        all_epochs.append(result['epochs'])
        all_losses.append(result['losses'])
        times.append(result['total_time'])
        final_losses.append(result['final_loss'])
    
    # Plot 1: Loss over Training
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (impl, epochs, losses) in enumerate(zip(implementations, all_epochs, all_losses)):
        ax1.plot(epochs, losses, marker='o', label=impl, 
                linewidth=2, markersize=4, color=colors[i])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see convergence better
    
    # Plot 2: Total Training Time
    ax2 = axes[1]
    x_pos = np.arange(len(implementations))
    bars = ax2.bar(x_pos, times, color=colors[:len(implementations)], alpha=0.7)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Total Training Time', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(implementations, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Speedup Comparison
    ax3 = axes[2]
    cuda_time = times[0]  # Assume first result is CUDA
    speedups = [cuda_time / t for t in times]
    bars = ax3.bar(x_pos, speedups, color=colors[:len(implementations)], alpha=0.7)
    ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Baseline (CUDA)')
    ax3.set_ylabel('Speedup (relative to CUDA)', fontsize=12)
    ax3.set_title('Performance Speedup', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(implementations, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(RESULTS_DIR, 'xor_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.show()


def main():
    """Main function to generate all plots"""
    print("Generating visualization plots...")
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        print("Please run benchmarks first")
        return
    
    # Plot MNIST results if available
    mnist_file = os.path.join(RESULTS_DIR, 'mnist_benchmark.json')
    if os.path.exists(mnist_file):
        print("\nPlotting MNIST results...")
        plot_mnist_results()
    else:
        print(f"\nSkipping MNIST plots - {mnist_file} not found")
    
    # Plot XOR results if available
    xor_file = os.path.join(RESULTS_DIR, 'xor_benchmark.json')
    if os.path.exists(xor_file):
        print("\nPlotting XOR results...")
        plot_xor_results()
    else:
        print(f"\nSkipping XOR plots - {xor_file} not found")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

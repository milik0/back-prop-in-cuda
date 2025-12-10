# CUDA MLP Benchmarks

This folder contains benchmarking tools to compare the performance of the CUDA MLP implementation with PyTorch.

## Overview

The benchmark suite compares:
- **CUDA Implementation**: Custom CUDA kernels for matrix operations and backpropagation
- **PyTorch Implementation**: Equivalent network using PyTorch's optimized operations

Four test cases are benchmarked:
1. **MNIST Classification**: 784 → 256 → 10 network on MNIST digit recognition
2. **Fashion-MNIST Classification**: 784 → 128 → 10 network on Fashion-MNIST clothing recognition
3. **Breast Cancer Wisconsin**: 30 → 16 → 8 → 1 network for binary classification
4. **XOR Problem**: 2 → 8 → 1 network on the classic XOR logic problem

## Setup

### 1. Install Python Dependencies

```bash
make benchmark-setup
```

Or manually:
```bash
pip3 install -r benchmark/requirements.txt
```

Required packages:
- `torch` (PyTorch)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

### 2. Build CUDA Executables

```bash
make                # Build MNIST executable
make xor            # Build XOR executable
make fashion-mnist  # Build Fashion-MNIST executable
make breast-cancer  # Build Breast Cancer executable
```

### 3. Download Datasets

#### MNIST Data

If you haven't already, download the MNIST dataset:

```bash
# Create data directory
mkdir -p ~/data

# Download and extract training data
wget -P ~/data/ https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -P ~/data/ https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
gzip -d ~/data/t10k-images-idx3-ubyte.gz
gzip -d ~/data/t10k-labels-idx1-ubyte.gz
```

#### Fashion-MNIST Data

```bash
# Create directory
mkdir -p ~/data/fashion_mnist

# Download and extract Fashion-MNIST
wget -P ~/data/fashion_mnist/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P ~/data/fashion_mnist/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P ~/data/fashion_mnist/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P ~/data/fashion_mnist/ http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# Decompress
gzip -d ~/data/fashion_mnist/*.gz
```

#### Breast Cancer Wisconsin Data

```bash
# Download from Kaggle or UCI ML Repository
# Option 1: From Kaggle (requires Kaggle account)
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p ~/data
unzip ~/data/breast-cancer-wisconsin-data.zip -d ~/data
mv ~/data/data.csv ~/data/breast_cancer.csv

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Download data.csv and save as ~/data/breast_cancer.csv
```

## Running Benchmarks

### Run All Benchmarks

```bash
make benchmark-all
```

This will:
1. Run MNIST benchmark (CUDA vs PyTorch)
2. Run Fashion-MNIST benchmark (CUDA vs PyTorch)
3. Run Breast Cancer benchmark (CUDA vs PyTorch)
4. Run XOR benchmark (CUDA vs PyTorch)
5. Generate visualization plots

### Run Individual Benchmarks

**MNIST Benchmark:**
```bash
make benchmark-mnist
```

**Fashion-MNIST Benchmark:**
```bash
make benchmark-fashion-mnist
```

**Breast Cancer Benchmark:**
```bash
make benchmark-breast-cancer
```
### Run Individual Benchmarks

**MNIST Benchmark:**
```bash
make benchmark-mnist
```

**XOR Benchmark:**
```bash
make benchmark-xor
```

**Generate Visualizations Only:**
```bash
make benchmark-visualize
```

### Manual Execution

You can also run the Python scripts directly:

```bash
cd benchmark

# MNIST benchmark
python3 benchmark_mnist.py

# XOR benchmark
python3 benchmark_xor.py

# Generate plots
python3 visualize.py
```

## Configuration

You can modify benchmark parameters by editing the respective Python files:

### `benchmark_mnist.py`
```python
DATA_PATH = os.path.expanduser("~/data")  # Path to MNIST data
EPOCHS = 5                                 # Number of training epochs
BATCH_SIZE = 64                           # Batch size
LEARNING_RATE = 0.01                      # Learning rate
```

### `benchmark_xor.py`
```python
EPOCHS = 10000                            # Number of training epochs
LEARNING_RATE = 0.01                      # Learning rate
```

## Output

### Results Files

Benchmark results are saved as JSON files in `benchmark/results/`:
- `mnist_benchmark.json`: MNIST benchmark results
- `xor_benchmark.json`: XOR benchmark results

### Visualization Plots

Generated plots are saved in `benchmark/results/`:
- `mnist_comparison.png`: MNIST performance comparison charts
- `xor_comparison.png`: XOR performance comparison charts

### Console Output

The benchmarks print a comparison table showing:
- **Implementation**: CUDA, PyTorch (cuda), PyTorch (cpu)
- **Time**: Total training time in seconds
- **Accuracy**: Training and test accuracy (MNIST) or final loss (XOR)
- **Speedup**: Relative speedup compared to CUDA baseline

Example output:
```
BENCHMARK COMPARISON SUMMARY
=================================================================
Implementation       Time (s)     Train Acc    Test Acc     Speedup   
----------------------------------------------------------------------
CUDA                 15.234       91.23        90.45        1.00x
PyTorch (cuda)       12.456       91.18        90.42        1.22x
```

## What Gets Measured

### Performance Metrics
- **Total Training Time**: Wall-clock time for complete training
- **Per-Epoch Time**: Time for each training epoch (MNIST)
- **Memory Usage**: GPU memory consumption (implicit in CUDA limits)

### Accuracy Metrics
- **Training Accuracy**: Accuracy on training batches (MNIST)
- **Test Accuracy**: Final accuracy on test set (MNIST)
- **Loss Convergence**: MSE loss over training (XOR)

## Understanding Results

### Expected Performance

**CUDA Implementation:**
- Lower-level control, potential for optimization
- Custom kernel implementations
- Minimal framework overhead
- May be slower initially without extensive optimization

**PyTorch Implementation:**
- Highly optimized CUDA kernels (cuBLAS, cuDNN)
- Automatic optimization and fusion
- Years of performance tuning
- Typically faster for standard operations

### Factors Affecting Performance

1. **Matrix Size**: Larger matrices benefit more from optimized libraries
2. **GPU Architecture**: Performance varies by GPU model
3. **Memory Transfer**: Data movement between CPU/GPU
4. **Kernel Optimization**: Quality of custom CUDA kernels
5. **Batch Size**: Affects parallelism and memory usage

## Files Structure

```
benchmark/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pytorch_mlp.py           # PyTorch model implementations
├── benchmark_mnist.py       # MNIST benchmarking script
├── benchmark_xor.py         # XOR benchmarking script
├── visualize.py             # Visualization generation
└── results/                 # Output directory
    ├── mnist_benchmark.json
    ├── xor_benchmark.json
    ├── mnist_comparison.png
    └── xor_comparison.png
```

## Troubleshooting

### CUDA Executable Not Found
```
Error: CUDA executable not found at ../bin/mlp_test
```
**Solution**: Run `make` or `make xor` to build the executables first.

### MNIST Data Not Found
```
Error: Data directory not found at ~/data
```
**Solution**: Download MNIST data using the commands in Setup section.

### PyTorch CUDA Not Available
```
Warning: CUDA not available, falling back to CPU
```
**Solution**: Install PyTorch with CUDA support or run on CPU (will be slower).

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Run `make benchmark-setup` to install dependencies.

## Customization

### Adding New Benchmarks

1. Create a new benchmark script in `benchmark/`
2. Import models from `pytorch_mlp.py`
3. Follow the pattern from existing benchmarks
4. Add a target to the Makefile
5. Update visualization script if needed

### Modifying Model Architecture

Edit `pytorch_mlp.py` to change:
- Layer sizes
- Activation functions
- Initialization schemes

Make corresponding changes to CUDA implementation in `src/` for fair comparison.

## Citation

If you use this benchmarking framework, please cite:
```
CUDA MLP Implementation
https://github.com/milik0/back-prop-in-cuda
```

## License

See main project LICENSE file.

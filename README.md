# back-prop-in-cuda
A complete implementation of a simple neural network backpropagation in CUDA, from scratch

to download data:
- train data:
```bash
# create data directory
mkdir -p data

# download train data of MNIST dataset
wget -P data/ https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -P data/ https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz

# decompress them (The code expects uncompressed binary files)
gzip -d data/train-images-idx3-ubyte.gz
gzip -d data/train-labels-idx1-ubyte.gz

# verify files are there
ls -l data/
```
- test data:
```bash
# download test data of MNIST dataset
wget -P data/ https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget -P data/ https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

# decompress them (The code expects uncompressed binary files)
gzip -d data/t10k-images-idx3-ubyte.gz
gzip -d data/t10k-labels-idx1-ubyte.gz

# verify files are there
ls -l data/
```
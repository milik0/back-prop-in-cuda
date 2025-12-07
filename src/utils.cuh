#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Matrix {
    float* data;
    int rows;
    int cols;
    bool allocated = false;

    void allocate(int r, int c) {
        if (allocated) free(); // Safety check
        rows = r;
        cols = c;
        CHECK_CUDA(cudaMalloc(&data, rows * cols * sizeof(float)));
        allocated = true;
    }

    void free() {
        if (allocated) {
            CHECK_CUDA(cudaFree(data));
            allocated = false;
        }
    }

    // Helper to zero out gradients
    void zeros() {
        if (allocated) CHECK_CUDA(cudaMemset(data, 0, rows * cols * sizeof(float)));
    }

    void copyFromHost(const std::vector<float>& hostData) {
        CHECK_CUDA(cudaMemcpy(data, hostData.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copyToHost(std::vector<float>& hostData) {
        hostData.resize(rows * cols);
        CHECK_CUDA(cudaMemcpy(hostData.data(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    }
};
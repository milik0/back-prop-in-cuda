#pragma once
#include "utils.cuh"
#include <fstream>
#include <vector>
#include <algorithm> // for std::reverse
#include <string>

// Helper to flip endianness (MNIST is Big Endian, Intel/Nvidia are Little Endian)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void loadMNIST(const std::string& image_path, const std::string& label_path, 
               Matrix& X, Matrix& Y, int num_samples) {
    
    // --- Load Images ---
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << image_path << std::endl;
        exit(1);
    }

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);
    int input_dim = n_rows * n_cols;

    // Allocate Host Memory
    std::vector<float> h_X(num_samples * input_dim);
    
    // Read pixel data
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            // Normalize 0-255 -> 0.0-1.0
            h_X[i * input_dim + j] = (float)temp / 255.0f;
        }
    }
    
    // Copy to GPU
    X.allocate(num_samples, input_dim);
    X.copyFromHost(h_X);
    file.close();

    // --- Load Labels (One-Hot Encoded) ---
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Error: Could not open " << label_path << std::endl;
        exit(1);
    }

    int label_magic = 0, number_of_labels = 0;
    label_file.read((char*)&label_magic, sizeof(label_magic));
    label_file.read((char*)&number_of_labels, sizeof(number_of_labels));

    int output_dim = 10; // digits 0-9
    std::vector<float> h_Y(num_samples * output_dim, 0.0f);

    for (int i = 0; i < num_samples; ++i) {
        unsigned char label = 0;
        label_file.read((char*)&label, sizeof(label));
        // One-Hot Encode
        if (label < 10) {
            h_Y[i * output_dim + label] = 1.0f;
        }
    }

    // Copy to GPU
    Y.allocate(num_samples, output_dim);
    Y.copyFromHost(h_Y);
    label_file.close();

    std::cout << "Loaded MNIST: " << num_samples << " samples." << std::endl;
}
#pragma once
#include <vector>
#include "layers.cuh"

class MLP {
    std::vector<Layer*> layers;

public:
    ~MLP() {
        for (auto layer : layers) delete layer;
    }

    void add(Layer* layer) {
        layers.push_back(layer);
    }

    Matrix forward(Matrix X) {
        Matrix current = X;
        for (auto layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    void backward(Matrix d_loss, float learning_rate) {
        Matrix current_grad = d_loss;
        // Iterate backwards
        for (int i = layers.size() - 1; i >= 0; --i) {
            current_grad = layers[i]->backward(current_grad, learning_rate);
        }
    }
};
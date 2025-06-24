#pragma once

#include "Tensor.hpp"

class FeedForward {
public:
    FeedForward(size_t embedding_dim, size_t hidden_dim) {
        // Inicializar las matrices de pesos para las capas
        W1 = Tensor({embedding_dim, hidden_dim});  // Capa 1: de embedding_dim a hidden_dim
        W2 = Tensor({hidden_dim, embedding_dim});  // Capa 2: de hidden_dim a embedding_dim

        // Inicializamos los pesos con valores pequeños (por ejemplo, ceros)
        W1.fill(0.01);
        W2.fill(0.01);
    }

    // Forward pass: Recibe la entrada y devuelve la salida de la red feed-forward
    Tensor forward(const Tensor &input) {
        // Aplicar la primera capa (W1)
        Tensor hidden = input * W1;  // [batch_size, sequence_length, hidden_dim]

        // Aplicar ReLU
        hidden = relu(hidden);  // [batch_size, sequence_length, hidden_dim]

        // Aplicar la segunda capa (W2)
        Tensor output = hidden * W2;  // [batch_size, sequence_length, embedding_dim]

        return output;
    }

private:
    Tensor W1, W2;  // Matrices de pesos
    size_t embedding_dim, hidden_dim;

    // Función de activación ReLU
    Tensor relu(const Tensor &input) {
        Tensor result(input.shape);
        for (size_t i = 0; i < input.get_size(); ++i) {
            result.data[i] = std::max(0.0f, input.data[i]);
        }
        return result;
    }
};

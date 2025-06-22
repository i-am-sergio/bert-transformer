#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

class PositionalEncoding {
private:
    size_t max_len;        // Longitud máxima de la secuencia
    size_t embedding_dim;  // Dimensión de los embeddings
    Tensor positional_encodings;  // Tensor de codificaciones posicionales

public:
    // Constructor: Inicializa las codificaciones posicionales
    PositionalEncoding(size_t max_len_, size_t embedding_dim_)
        : max_len(max_len_), embedding_dim(embedding_dim_) {

        // Inicializa el tensor de codificaciones posicionales con la forma [max_len, embedding_dim]
        positional_encodings = Tensor({max_len, embedding_dim});
        initialize_positional_encodings();
    }

    // Inicialización de las codificaciones posicionales usando funciones trigonométricas
    void initialize_positional_encodings() {
        for (size_t pos = 0; pos < max_len; ++pos) {
            for (size_t i = 0; i < embedding_dim; ++i) {
                // Aplicamos la fórmula de codificación posicional (seno/coseno)
                float angle = static_cast<float>(pos) / pow(10000, (2 * (i / 2)) / static_cast<float>(embedding_dim));
                if (i % 2 == 0) {
                    positional_encodings({pos, i}) = sin(angle);  // Para las posiciones pares
                } else {
                    positional_encodings({pos, i}) = cos(angle);  // Para las posiciones impares
                }
            }
        }
    }

    // Forward pass: Añadir las codificaciones posicionales a los embeddings
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];   // Tamaño del lote
        size_t seq_len = input.shape[1];      // Longitud de la secuencia

        // Creamos un tensor de salida con la forma [batch_size, seq_len, embedding_dim]
        Tensor output({batch_size, seq_len, embedding_dim});

        // Añadimos las codificaciones posicionales a los embeddings de entrada
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t k = 0; k < embedding_dim; ++k) {
                    output({i, j, k}) = input({i, j, k}) + positional_encodings({j, k});
                }
            }
        }

        return output;
    }

    // Función para imprimir las codificaciones posicionales (para depuración)
    void print_positional_encodings() const {
        for (size_t i = 0; i < max_len; ++i) {
            for (size_t j = 0; j < embedding_dim; ++j) {
                cout << positional_encodings({i, j}) << " ";
            }
            cout << endl;
        }
    }
};

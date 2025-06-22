#pragma once

#include "Tensor.hpp"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class LayerNormalization {
private:
    size_t embedding_dim;  // Dimensión de los embeddings
    Tensor gamma;          // Parámetro de escala
    Tensor beta;           // Parámetro de desplazamiento
    float epsilon;         // Valor pequeño para evitar la división por cero

public:
    // Constructor: Inicializa la normalización de capa
    LayerNormalization(size_t embedding_dim_, float epsilon_ = 1e-6f)
        : embedding_dim(embedding_dim_), epsilon(epsilon_) {
        // Inicializamos los parámetros gamma y beta (usados para la escala y el desplazamiento)
        gamma = Tensor({embedding_dim});
        beta = Tensor({embedding_dim});

        // Inicializamos gamma a 1 y beta a 0
        gamma.fill(1.0f);
        beta.fill(0.0f);
    }

    // Forward pass: Aplica la normalización de capa
    Tensor forward(const Tensor &input) {
        size_t batch_size = input.shape[0];  // Tamaño del lote
        size_t seq_len = input.shape[1];     // Longitud de la secuencia

        // Calculamos la media y la varianza por cada ejemplo en el lote
        Tensor mean({batch_size, seq_len});
        Tensor var({batch_size, seq_len});

        // Calcular la media
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float sum = 0;
                for (size_t d = 0; d < embedding_dim; ++d) {
                    sum += input({b, s, d});
                }
                mean({b, s}) = sum / embedding_dim;
            }
        }

        // Calcular la varianza
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float sum_sq = 0;
                for (size_t d = 0; d < embedding_dim; ++d) {
                    sum_sq += pow(input({b, s, d}) - mean({b, s}), 2);
                }
                var({b, s}) = sum_sq / embedding_dim;
            }
        }

        // Normalizar la entrada
        Tensor output({batch_size, seq_len, embedding_dim});
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < embedding_dim; ++d) {
                    float normalized_value = (input({b, s, d}) - mean({b, s})) /
                                             sqrt(var({b, s}) + epsilon);
                    output({b, s, d}) = gamma({d}) * normalized_value + beta({d});
                }
            }
        }

        return output;
    }
};

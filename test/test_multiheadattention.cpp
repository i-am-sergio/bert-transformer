#include "MultiHeadAttention.hpp"
#include "Tensor.hpp"
#include <iostream>

int main() {
    // Parámetros
    size_t batch_size = 2;       // Tamaño del lote
    size_t seq_len = 5;          // Longitud de la secuencia
    size_t embedding_dim = 16;   // Dimensión de los embeddings
    size_t num_heads = 4;        // Número de cabezas de atención

    // Crear la capa de atención múltiple
    MultiHeadAttention attention(num_heads, embedding_dim);

    // Crear un tensor de entrada de ejemplo (batch_size, seq_len, embedding_dim)
    Tensor input({batch_size, seq_len, embedding_dim});

    // Inicializar el tensor de entrada con algunos valores
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            for (size_t k = 0; k < embedding_dim; ++k) {
                input({i, j, k}) = static_cast<float>(i + j + k);  // Valores arbitrarios para la prueba
            }
        }
    }

    // Imprimir el tensor de entrada
    cout << "Entrada (Tensor de Embeddings):" << endl;
    cout << input << endl;

    // Realizar la atención múltiple
    Tensor output = attention.forward(input);

    // Imprimir la salida
    cout << "Salida de la Atención Múltiple:" << endl;
    cout << output << endl;

    return 0;
}
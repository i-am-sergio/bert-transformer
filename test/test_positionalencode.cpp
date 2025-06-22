#include "PositionalEncoding.hpp"
#include "Tensor.hpp"
#include <iostream>

int main() {
    size_t max_len = 10;         // Longitud máxima de la secuencia
    size_t embedding_dim = 300;   // Dimensión de los embeddings

    // Crear la capa de codificación posicional
    PositionalEncoding pos_encoding(max_len, embedding_dim);

    // Crear un tensor de entrada de ejemplo (embeddings)
    Tensor input({2, 5, embedding_dim});  // Lote de 2, secuencia de 5 tokens
    input.fill(1.0f);  // Llenamos con valores arbitrarios

    // Obtener las salidas sumando los embeddings con las codificaciones posicionales
    Tensor output = pos_encoding.forward(input);
    cout << "Salida después de sumar codificaciones posicionales: " << endl;
    cout << output << endl;

    return 0;
}

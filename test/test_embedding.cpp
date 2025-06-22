#include "EmbeddingLayer.hpp"
#include "Tensor.hpp"
#include <iostream>

int main() {
    size_t vocab_size = 10000; // Suponiendo un vocabulario de tamaño 10000
    size_t embedding_dim = 300; // Usando embeddings de 300 dimensiones

    // Crear la capa de embeddings
    EmbeddingLayer embedding_layer(vocab_size, embedding_dim);

    // Crear un tensor de entrada de ejemplo (índices de tokens)
    Tensor input({2, 5});  // Lote de 2, secuencia de 5 tokens
    input({0, 0}) = 1; input({0, 1}) = 2; input({0, 2}) = 3; input({0, 3}) = 4; input({0, 4}) = 5;
    input({1, 0}) = 2; input({1, 1}) = 3; input({1, 2}) = 4; input({1, 3}) = 5; input({1, 4}) = 6;

    // Obtener los embeddings correspondientes
    Tensor embeddings = embedding_layer.forward(input);
    cout << "Embeddings generados: " << endl;
    cout << embeddings << endl;  // Imprimir los embeddings

    // Supongamos que tenemos gradientes de la capa posterior (simulación)
    Tensor grad_output({2, 5, embedding_dim});
    grad_output.fill(0.1f);  // Llenamos con valores de gradiente arbitrarios

    // Calcular los gradientes de los embeddings
    Tensor grad_embeddings = embedding_layer.backward(grad_output, input);

    // Actualizar los embeddings (simulación de optimización)
    float learning_rate = 0.01f;
    embedding_layer.update_embeddings(grad_embeddings, learning_rate);

    return 0;
}

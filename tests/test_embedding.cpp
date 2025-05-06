#include "layers/EmbeddingLayer.hpp"
#include "core/Tensor.hpp"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

int main() {
    // Definir parámetros de prueba
    size_t vocabulary_size = 10;
    size_t embedding_dimension = 5;
    EmbeddingLayer embedding_layer(vocabulary_size, embedding_dimension);

    // --- Prueba 1: Forward pass con un solo índice ---
    cout << "--- Prueba 1: Forward pass con un solo índice ---" << endl;
    Tensor input_indices_single({1}, {2.0f}); // Un solo índice (token ID)
    Tensor output_embedding_single = embedding_layer.forward({input_indices_single});
    cout << "Forma de la salida: ";
    for (size_t dim : output_embedding_single.getShape()) {
        cout << dim << " ";
    }
    cout << endl;
    assert(output_embedding_single.getShape().size() == 2);
    assert(output_embedding_single.getShape()[0] == 1);
    assert(output_embedding_single.getShape()[1] == embedding_dimension);
    cout << "Prueba 1 completada." << endl;

    // --- Prueba 2: Forward pass con una secuencia de índices ---
    cout << "\n--- Prueba 2: Forward pass con una secuencia de índices ---" << endl;
    Tensor input_indices_sequence({3}, {0.0f, 5.0f, 9.0f}); // Una secuencia de 3 tokens
    Tensor output_embedding_sequence = embedding_layer.forward({input_indices_sequence});
    cout << "Forma de la salida: ";
    for (size_t dim : output_embedding_sequence.getShape()) {
        cout << dim << " ";
    }
    cout << endl;
    assert(output_embedding_sequence.getShape().size() == 2);
    assert(output_embedding_sequence.getShape()[0] == 3);
    assert(output_embedding_sequence.getShape()[1] == embedding_dimension);
    cout << "Prueba 2 completada." << endl;

    // --- Prueba 3: Forward pass con un batch de secuencias ---
    cout << "\n--- Prueba 3: Forward pass con un batch de secuencias ---" << endl;
    Tensor input_indices_batch({2, 4}, {1.0f, 3.0f, 7.0f, 0.0f, 8.0f, 2.0f, 5.0f, 9.0f}); // Batch de 2 secuencias de longitud 4
    Tensor output_embedding_batch = embedding_layer.forward({input_indices_batch});
    cout << "Forma de la salida: ";
    for (size_t dim : output_embedding_batch.getShape()) {
        cout << dim << " ";
    }
    cout << endl;
    assert(output_embedding_batch.getShape().size() == 3);
    assert(output_embedding_batch.getShape()[0] == 2);
    assert(output_embedding_batch.getShape()[1] == 4);
    assert(output_embedding_batch.getShape()[2] == embedding_dimension);
    cout << "Prueba 3 completada." << endl;

    // --- Prueba 4: Índice fuera de rango (debería lanzar una excepción) ---
    cout << "\n--- Prueba 4: Índice fuera de rango ---" << endl;
    Tensor invalid_input({1}, {15.0f}); // Índice mayor que el tamaño del vocabulario
    try {
        embedding_layer.forward({invalid_input});
        cerr << "Error: No se lanzó excepción para índice fuera de rango." << endl;
        return 1;
    } catch (const out_of_range& e) {
        cout << "Excepción capturada correctamente: " << e.what() << endl;
    }
    cout << "Prueba 4 completada." << endl;

    // --- Prueba 5: Backward pass (verificar formas de los gradientes) ---
    cout << "\n--- Prueba 5: Backward pass (verificar formas de los gradientes) ---" << endl;
    Tensor output_gradient_batch(output_embedding_batch.getShape()); // Gradiente de la misma forma que la salida
    Tensor input_gradient = embedding_layer.backward(output_gradient_batch, {input_indices_batch});
    cout << "Forma del gradiente de entrada: ";
    for (size_t dim : input_gradient.getShape()) {
        cout << dim << " ";
    }
    cout << endl;
    assert(input_gradient.getShape() == input_indices_batch.getShape());
    cout << "Prueba de forma del gradiente de entrada completada." << endl;


    cout << "\n¡Todas las pruebas básicas de la EmbeddingLayer completadas!" << endl;

    return 0;
}
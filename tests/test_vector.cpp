#include <iostream>
#include <vector>
#include "bert.hpp"
#include "core/Tensor.hpp"

int main() {
    BERT bert_instance;
    bert_instance.printMessage();


    // Crear una matriz 2x3 inicializada con ceros
    Tensor matrix1({2, 3});
    matrix1.print();

    // Crear una matriz 2x2 con datos iniciales
    Tensor matrix2({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    matrix2.print();

    // Acceder y modificar elementos
    std::cout << "Elemento en (0, 1) de matrix2: " << matrix2.at({0, 1}) << std::endl;
    matrix2.at({1, 0}) = 9.9f;
    matrix2.print();

    // Intentar acceder a un índice fuera de rango (esto lanzará una excepción)
    try {
        std::cout << matrix1.at({2, 0}) << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

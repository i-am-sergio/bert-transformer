#pragma once 

#include "Tensor.hpp" // Incluye la definición de la clase Tensor

#include <cmath>      // Funciones matemáticas estándar
#include <cassert>    // Para aserciones (debug)
#include <stdexcept>  // Para manejo de errores
#include <omp.h>      // Biblioteca para paralelismo con OpenMP

// Función para multiplicación de tensores (producto punto generalizado)
Tensor dot_product(const Tensor &a, const Tensor &b) {
    const auto& a_data = a.get_data();      // Obtiene los datos del tensor 'a'
    const auto& b_data = b.get_data();      // Obtiene los datos del tensor 'b'
    const auto& b_shape = b.get_shape();    // Obtiene la forma (dimensiones) de 'b'
    size_t N = b_shape[0]; // input_dim     // Número de entradas (filas de 'b')
    size_t M = b_shape[1]; // output_dim    // Número de salidas (columnas de 'b')

    Tensor result({M});                     // Crea un tensor resultado con M elementos
    auto& r_data = result.get_data();       // Referencia a los datos del resultado

    #pragma omp parallel for                // Paraleliza el siguiente bucle con OpenMP
    for (size_t i = 0; i < M; i++) {        // Itera sobre cada dimensión de salida
        float sum = 0.0f;                   // Acumulador para el resultado de la suma
        for (size_t j = 0; j < N; j++) {    // Itera sobre la dimensión de entrada
            sum += a_data[j] * b_data[j * M + i]; // Multiplica y acumula (producto punto)
        }
        r_data[i] = sum;                    // Guarda el resultado en la posición i
    }
    return result;                          // Retorna el tensor resultante
}
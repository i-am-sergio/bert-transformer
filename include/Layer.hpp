#pragma once
#include "Tensor.hpp"
#include "Math.hpp"
#include "Optimizer.hpp"

#include <random>
#include <algorithm>
#include <cmath>

class Layer { // Clase base abstracta para capas de red neuronal
    public:
    virtual ~Layer() = default; // Destructor virtual por defecto

    virtual Tensor forward(const Tensor& input) = 0;      // Propagacion hacia adelante (funcion pura)
    virtual Tensor backward(const Tensor& grad_output) = 0; // Propagacion hacia atras (funcion pura)
    virtual void update_parameters(Optimizer& optimizer) = 0; // Actualizar pesos (funcion pura)
    virtual void print() const = 0;                     // Imprimir capa 
    virtual void zero_grad() = 0;
};



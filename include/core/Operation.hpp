#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "Tensor.hpp" // Asegúrate de que la ruta sea correcta
#include <vector>

class Operation {
public:
  virtual ~Operation() = default; // Destructor virtual para herencia

  // Realiza la operación hacia adelante
  virtual Tensor forward(const std::vector<Tensor> &inputs) = 0;

  // Calcula el gradiente de la operación con respecto a sus entradas
  virtual Tensor backward(const Tensor &outputGradient,
                          const std::vector<Tensor> &inputs) = 0;
};

#endif // OPERATION_HPP

#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "Tensor.hpp" // Asegúrate de que la ruta sea correcta
#include <vector>

class Operation {
public:
  virtual ~Operation() = default; // Destructor virtual para herencia

  // Realiza la operación hacia adelante
  virtual Tensor forward(const std::vector<Tensor> &inputs) = 0;

  // Calcula el gradiente de la operación con respecto a sus entradas
  virtual Tensor backward(const Tensor &outputGradient,
                          const std::vector<Tensor> &inputs) = 0;
};

#endif // OPERATION_HPP
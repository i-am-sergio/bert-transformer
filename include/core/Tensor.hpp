#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

class Tensor
{
private:
    vector<size_t> shape_; // Forma del tensor (ejemplo: {2, 3} para una matriz 2x3)
    vector<float> data_; // Datos del tensor (almacenados en un vector plano)
    vector<size_t> strides_; // Strides para acceder a los datos eficientemente

    // Calcula los strides para acceder eficientemente a los elementos
    void calculateStrides()
    {
        strides_.resize(shape_.size()); // Inicializa el vector de strides
        if (shape_.empty())
            return;          // Si no hay dimensiones, no hay strides
        strides_.back() = 1; // El stride de la última dimensión siempre es 1
        for (size_t i = shape_.size() - 1; i > 0; --i)
        {                                              // Recorre las dimensiones de atrás hacia adelante
            strides_[i - 1] = strides_[i] * shape_[i]; // Calcula el stride de la dimensión anterior
        }
    }

    // Función recursiva para imprimir el tensor de forma legible
    void printRecursive(vector<size_t> current_indices, size_t dimension) const
    {
        if (dimension == shape_.size())
        {
            cout << at(current_indices) << " ";
            return;
        }

        for (size_t i = 0; i < shape_[dimension]; ++i)
        {
            current_indices[dimension] = i;
            printRecursive(current_indices, dimension + 1);
            if (dimension == shape_.size() - 1)
            {
                // No newline para la última dimensión de un sub-tensor
            }
            else if (i == shape_[dimension] - 1)
            {
                cout << endl;
                for (size_t j = 0; j < dimension; ++j)
                {
                    cout << "  "; // Indentación para mostrar la estructura
                }
            }
        }
    }

public:
    // Constructor que recibe la forma del tensor (ej: {2, 3} para una matriz 2x3)
    Tensor(const vector<size_t> &shape)
    {
        this->shape_ = shape; // Inicializa la forma del tensor
        size_t total_elements = 1;
        for (size_t dim_size : shape_)
        {
            total_elements *= dim_size;
        }
        data_.resize(total_elements, 0.0f); // Inicializa con ceros
        calculateStrides();                 // Calcula los strides para acceder eficientemente a los datos
    }

    // Constructor que recibe la forma y los datos iniciales
    Tensor(const vector<size_t>& shape, const vector<float>& data) {
        this->shape_ = shape; // Inicializa la forma del tensor
        this->data_ = data;   // Inicializa los datos del tensor
        size_t total_elements = 1;
        for (size_t dim_size : shape_)
        { // Calcula el número total de elementos
            total_elements *= dim_size;
        }
        if (data_.size() != total_elements)
        {
            throw invalid_argument("El tamaño de los datos no coincide con la forma del tensor.");
        }
        calculateStrides();
    }

    // Devuelve la forma (dimensiones) del tensor
    const vector<size_t> &getShape() const
    {
        return shape_;
    }

    // Devuelve los datos del tensor (como un vector plano)
    const vector<float>& getData() const {
        return data_;
    }

    // Accede al elemento del tensor usando un vector de índices. Ejemplo com matriz 3 x 5 x 2
    float at(const vector<size_t>& indices) const { 
        if (indices.size() != shape_.size()) {
            throw out_of_range("Número incorrecto de índices.");
        }
        size_t offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            if (indices[i] >= shape_[i])
            { // Verifica si el índice está dentro de los límites
                throw out_of_range("Índice fuera de rango en la dimensión " + to_string(i) + ".");
            }
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }

    // Accede al elemento del tensor usando un vector de índices (para modificar)
    float& at(const vector<size_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw out_of_range("Número incorrecto de índices.");
        }
        size_t offset = 0;
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            if (indices[i] >= shape_[i])
            {
                throw out_of_range("Índice fuera de rango en la dimensión " + to_string(i) + ".");
            }
            offset += indices[i] * strides_[i];
        }
        return data_[offset];
    }

    // Imprime el contenido del tensor (para debugging)
    void print() const
    {
        printRecursive(vector<size_t>(shape_.size(), 0), 0);
        cout << endl;
    }
};
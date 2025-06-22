#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace std;

class Tensor {
public:
    vector<size_t> shape;   // Dimensiones del tensor (ej: [2,3] = matriz 2x3)
    vector<size_t> strides; // Pasos para navegar entre elementos en memoria
    vector<float> data;     // Datos almacenados en un array lineal

    // Constructor vacio
    Tensor() {}

    // Constructor con forma especifica
    Tensor(const vector<size_t> &shape_) : shape(shape_) {
        size_t total_size = 1;
        for (auto dim : shape)
            total_size *= dim;
        data.resize(total_size);
        compute_strides();
    }

    Tensor(initializer_list<size_t> shape_) : shape(shape_) {
        size_t total_size = 1;
        for (auto dim : shape)
            total_size *= dim;
        data.resize(total_size);
        compute_strides();
    }

    // Acceso a elementos mediante indices (version modificable)
    float &operator()(const vector<size_t> &indices) {
        return data[compute_offset(indices)];
    }

    // Acceso a elementos mediante indices (version constante)
    const float &operator()(const vector<size_t> &indices) const {
        return data[compute_offset(indices)];
    }

    // Rellena todo el tensor con un valor
    void fill(float value) { std::fill(data.begin(), data.end(), value); }

    // Imprime el tensor con formato segun su dimensionalidad
    void print() const { print_recursive(0, 0); cout << endl; }

    // Devuelve el numero total de elementos en el tensor
    size_t get_size() const { return data.size(); }

    // Transposición de matrices 2D
    Tensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("La transposicion solo para tensores 2D.");
        }

        size_t rows = shape[0];
        size_t cols = shape[1];
        Tensor transposed({cols, rows});

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed({j, i}) = (*this)({i, j});
            }
        }

        return transposed;
    }

    // Sobrecarga del operador << para imprimir el tensor
    friend ostream &operator<<(ostream &os, const Tensor &tensor) {
        streambuf* old_buf = cout.rdbuf();
        cout.rdbuf(os.rdbuf());
        tensor.print_recursive(0, 0);
        cout.rdbuf(old_buf);
        return os;
    }

private:
    // Calcula la posicion en el array lineal dado un conjunto de indices
    size_t compute_offset(const vector<size_t> &indices) const {
        assert(indices.size() == shape.size());
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (indices[i] >= shape[i])
                throw out_of_range("Index out of bounds");
            offset += strides[i] * indices[i];
        }
        return offset;
    }

    // Calcula los pasos (strides) para navegar entre dimensiones
    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // Imprime el tensor recursivamente con indentacion para formato
    void print_recursive(size_t dim, size_t offset, size_t indent = 0) const {
        if (dim == shape.size() - 1) {
            cout << string(indent, ' ') << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                cout << data[offset + i];
                if (i + 1 < shape[dim])
                    cout << ", ";
            }
            cout << "]";
        } else {
            cout << string(indent, ' ') << "[\n";
            for (size_t i = 0; i < shape[dim]; ++i) {
                print_recursive(dim + 1, offset + i * strides[dim], indent + 2);
                if (i + 1 < shape[dim])
                    cout << ",\n";
            }
            cout << "\n" << string(indent, ' ') << "]";
        }
    }
};

// Imprimir vectores de cualquier tipo 
template<typename T>
inline ostream& operator<<(ostream& os, const vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i + 1 < vec.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Sobrecarga del operador "+" para sumar dos tensores elemento por elemento
Tensor operator+(const Tensor &tensorA, const Tensor &tensorB) {
    if (tensorA.shape != tensorB.shape) {
        throw invalid_argument("Tensors must have the same shape for addition.");
    }
    
    Tensor result(tensorA.shape);

    #pragma omp parallel for
    for (size_t i = 0; i < tensorA.data.size(); ++i) {
        result.data[i] = tensorA.data[i] + tensorB.data[i];
    }

    return result;
}
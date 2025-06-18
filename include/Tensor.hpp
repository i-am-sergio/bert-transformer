#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace std;

class Tensor {
private:
    vector<size_t> shape;   // Forma del tensor (dimensiones)
    vector<size_t> strides; // Saltos necesarios para acceder a un index
    vector<float> data;     // Datos almacenados en forma lineal

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

    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

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

public:
    Tensor() {}
    Tensor(const vector<size_t> &shape_) : shape(shape_) {
        size_t total_size = 1;
        for (auto dim : shape)
            total_size *= dim;
        data.resize(total_size);
        compute_strides();
    }

    float &operator()(const vector<size_t> &indices) {
        return data[compute_offset(indices)];
    }

    const float &operator()(const vector<size_t> &indices) const {
        return data[compute_offset(indices)];
    }

    const vector<size_t> &get_shape() const { return shape; }
    const vector<float> &get_data() const { return data; }
    vector<float> &get_data() { return data; }

    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }

    void printLinear() const {
        for (const auto &val : data)
            cout << val << " ";
        cout << endl;
    }

    void print() const {
        print_recursive(0, 0);
        cout << endl;
    }

    size_t get_size() const {
        return data.size();
    }

    // Sobrecarga del operador << como friend
    friend ostream &operator<<(ostream &os, const Tensor &tensor) {
        // Guardamos y redirigimos temporalmente cout al ostream dado
        streambuf* old_buf = cout.rdbuf();
        cout.rdbuf(os.rdbuf());
        tensor.print_recursive(0, 0);  // usa print_recursive directamente
        cout.rdbuf(old_buf);  // Restauramos cout
        return os;
    }
};

// AdamOptimizer.hpp
#pragma once
#include <vector>
#include <cmath>

class AdamOptimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    std::vector<float> m; // Primer momento (media)
    std::vector<float> v; // Segundo momento (varianza no centrada)
    std::vector<float> params; // Referencia a los parámetros del modelo

public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    // Método para inicializar con los parámetros del modelo
    void initialize(const std::vector<float>& parameters) {
        params = parameters;
        m.resize(params.size(), 0.0f);
        v.resize(params.size(), 0.0f);
        t = 0;
    }

    // Método step para actualizar los parámetros
    void step(std::vector<float>& parameters, const std::vector<float>& gradients) {
        t++;
        
        // Tamaño de los parámetros debe coincidir con los momentos
        if (m.size() != parameters.size()) {
            initialize(parameters);
        }

        for (size_t i = 0; i < parameters.size(); ++i) {
            // Actualizar momentos
            m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * gradients[i] * gradients[i];
            
            // Corrección de bias
            float m_hat = m[i] / (1.0f - std::pow(beta1, t));
            float v_hat = v[i] / (1.0f - std::pow(beta2, t));
            
            // Actualizar parámetros
            parameters[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    // Método step alternativo para compatibilidad
    void update(std::vector<float>& params, const std::vector<float>& grads) {
        step(params, grads);
    }
};
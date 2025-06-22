#include "Tensor.hpp"
#include <cmath>


Tensor batch_matmul(const Tensor &A, const Tensor &B, bool transpose_B = false) {
    auto A_shape = A.shape; // [batch, M, K]
    auto B_shape = B.shape; // [batch, N, K] si transpuesta -> [batch, K, N]

    size_t batch = A_shape[0];
    size_t M = A_shape[1];
    size_t K = A_shape[2];
    size_t N = transpose_B ? B_shape[1] : B_shape[2];

    Tensor result({batch, M, N});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    float a = A({b, i, k});
                    float b_val = transpose_B ? B({b, j, k}) : B({b, k, j});
                    sum += a * b_val;
                }
                result({b, i, j}) = sum;
            }
        }
    }
    return result;
}



Tensor softmax(const Tensor& input) {
    auto shape = input.shape; // [batch, seq_len, seq_len]
    Tensor output(shape);

    size_t batch = shape[0];
    size_t rows = shape[1];
    size_t cols = shape[2];

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            float max_val = -INFINITY;
            for (size_t j = 0; j < cols; ++j)
                max_val = max(max_val, input({b, i, j}));

            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                float val = exp(input({b, i, j}) - max_val); // para estabilidad
                output({b, i, j}) = val;
                sum += val;
            }
            for (size_t j = 0; j < cols; ++j)
                output({b, i, j}) /= sum;
        }
    }

    return output;
}


Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    size_t d_k = Q.shape[2];
    float scale = 1.0f / sqrt(d_k);

    Tensor scores = batch_matmul(Q, K, true); // Q * K^T
    for (auto& val : scores.data)
        val *= scale;

    Tensor probs = softmax(scores);
    Tensor output = batch_matmul(probs, V); // softmax(QKᵀ / √dk) * V
    return output;
}

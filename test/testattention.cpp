#include "Tensor.hpp"
#include "Attention.hpp"

int main() {
    size_t batch = 1, seq_len = 4, d_k = 8, d_v = 8;

    Tensor Q({batch, seq_len, d_k});
    Tensor K({batch, seq_len, d_k});
    Tensor V({batch, seq_len, d_v});

    // Rellenar con valores aleatorios o fijos para probar
    for (auto& x : Q.get_data()) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : K.get_data()) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : V.get_data()) x = static_cast<float>(rand()) / RAND_MAX;

    Tensor result = attention(Q, K, V);
    cout << "Resultado de la atención:\n" << result << endl;

    return 0;
}

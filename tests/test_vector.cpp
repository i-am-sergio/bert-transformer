#include <iostream>
#include <vector>
#include "bert.hpp"

int main() {
    BERT bert_instance;
    std::vector<int> A = {1, 2, 3, 4, 5};
    std::vector<int> B = {6, 7, 8, 9, 10};
    for (int i = 0; i < A.size(); i++) {
        std::cout << "Number: " << A[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < B.size(); i++) {
        std::cout << "Number: " << B[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Suma:\n";
    for (int i = 0; i < A.size(); i++) {
        std::cout << A[i] + B[i] << " ";
    }
    return 0;
}

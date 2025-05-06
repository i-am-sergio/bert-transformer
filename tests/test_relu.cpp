#include <iostream>
#include <vector>
#include "bert.hpp"
#include "core/Tensor.hpp"
#include "core/Activation.hpp"

int main()
{
    BERT bert_instance;
    bert_instance.printMessage();

    std::cout << "--- Prueba de ReLU (forward) ---" << std::endl;
    Tensor input_relu({2, 3}, {-1.0f, 2.0f, -3.0f, 0.0f, 4.5f, -0.5f});
    std::cout << "Tensor de entrada:" << std::endl;
    input_relu.print();
    ReLU relu_op;
    std::vector<Tensor> relu_input = {input_relu};
    Tensor output_relu = relu_op.forward(relu_input);
    std::cout << "Tensor de salida después de ReLU:" << std::endl;
    output_relu.print();

    std::cout << "---------------------------------" << std::endl;

    std::cout << "--- Prueba de ReLU (backward) ---" << std::endl;
    Tensor output_gradient_relu({2, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    std::cout << "Gradiente de salida:" << std::endl;
    output_gradient_relu.print();
    Tensor input_gradient_relu = relu_op.backward(output_gradient_relu, relu_input);
    std::cout << "Gradiente de entrada después de ReLU (backward):" << std::endl;
    input_gradient_relu.print();

    return 0;
}
#include "Conv2D.hpp"
#include "Dropout.hpp"
#include "Flatten.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "Reader.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"

#include <iostream>

// Hiperparametros
const int EPOCHS = 3;
const float LEARNING_RATE = 0.001f;
const string LOSS_FUNCTION = "cross-entropy";
const string OPTIMIZER = "sgd"; // adam / rmsprop 
const int BATCH_SIZE = 10;

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test);

int main() {
    NeuralNetwork model;

    // Arquitectura
    model.add_layer(flatten());
    model.add_layer(dense(784, 72, "relu", 0.001));
    model.add_layer(dense(72, 48, "relu", 0.001));
    model.add_layer(dense(48, 10, "softmax", 0.001));

    model.compile(LOSS_FUNCTION, OPTIMIZER, LEARNING_RATE);

    // Cargar datos (dataset MNIST) en tensores 4D para X y 1D para Y
    auto [X_all, Y_all] = Reader::load_bin("../topicos-inteligencia-artificial/datasets/MNIST/mnist_train.bin", 60000);
    auto [X_test, Y_test] = Reader::load_bin("../topicos-inteligencia-artificial/datasets/MNIST/mnist_test.bin", 10000);

    // Dividir en entrenamiento y validacion (80% train, 20% validation)
    auto [X_train, Y_train, X_val, Y_val] = train_test_split(X_all, Y_all, 0.2f, 37);
    cout << "- test data: " << X_test.size() << " -> " << X_test[0].shape << endl;
    
    // Train Model
    auto start = start_timer();
    model.fit(X_train, Y_train, X_val, Y_val, EPOCHS, BATCH_SIZE, 1, true);
    double duration = stop_timer(start);
    print_duration(duration, "Tiempo de entrenamiento");

    // Save Model
    cout << "\nGuardando el modelo entrenado..." << endl;
    model.save_model("mlp_mnist.bin");

    test_model(model, X_test, Y_test);

    return 0;
}

void test_model(NeuralNetwork &model, vector<Tensor> &X_test, vector<Tensor> &Y_test) {
    int correct = 0;
    int total = X_test.size();

    for (int i = 0; i < total; i++) {
        Tensor pred = model.predict(X_test[i]);
        int pred_label = argmax(pred);
        int true_label = argmax(Y_test[i]);

        if (pred_label == true_label)
            correct++;
    }

    float accuracy = 100.0f * correct / total;
    cout << "Precision en test: " << fixed << setprecision(2) << accuracy << "%" << endl;
}
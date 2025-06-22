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

    // // Arquitectura CNN con Maxpooling
    // model.add_layer(conv2d(1, 8, 5, 2, 2));        // 8 filtros 5x5, stride=2 -> [8,14,14]
    // model.add_layer(pool(2, 2, PoolingType::MAX)); // [8, 7, 7]
    // model.add_layer(flatten());                    // [8×7×7 = 392]
    // model.add_layer(dense(392, 128, "relu"));
    // model.add_layer(dense(128, 10, "softmax"));

    // Arquitectura CNN sin pooling
    model.add_layer(conv2d(1, 8, 3, 2, 1));         // 8 filtros, kernel 3x3, stride=1, pad=1
    model.add_layer(flatten());                    // [8×14×14 = 1568]
    model.add_layer(dense(1568, 64, "relu"));
    model.add_layer(dense(64, 10, "softmax"));

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
    model.save_model("cnn_mnist.bin");

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
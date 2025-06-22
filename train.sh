#!/bin/bash

if [ "$1" == "mlp" ]; then
  g++ -fopenmp -O3 -std=c++17 mlp.cpp -Iinclude -o mlp && ./mlp
elif [ "$1" == "cnn" ]; then
  g++ -fopenmp -O3 -std=c++17 cnn.cpp -Iinclude -o cnn && ./cnn
elif [ "$1" == "testcnn" ]; then
  g++ -fopenmp -O3 -std=c++17 test.cpp -Iinclude -o testcnn && ./testcnn
elif [ "$1" == "bert" ]; then
  g++ -fopenmp -O3 -std=c++17 bert.cpp -Iinclude -o bert && ./bert
else
  echo "Uso: $0 [mlp|cnn|testcnn|bert]"
  exit 1
fi
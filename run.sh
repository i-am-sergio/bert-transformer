# Clean previous build
if [ -d build ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

mkdir build
cd build
cmake ..
make -j4

# Run the program
./bert_test
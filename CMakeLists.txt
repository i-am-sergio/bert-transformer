# Versión mínima requerida de CMake
cmake_minimum_required(VERSION 3.15)

# Nombre del proyecto
project(bert-cpp
    VERSION 1.0.0
    DESCRIPTION "Implementación de BERT desde cero en C++"
    LANGUAGES CXX
)

# Configuración estándar de C++ (requiere C++17 o superior)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Usar estándar ISO (sin extensiones de compilador)

# Opciones de compilación (ajustar según necesidades)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")  # Habilitar warnings
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")            # Optimizaciones para release
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")            # Símbolos de debug

# Configurar rutas principales
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_DIR ${ROOT_DIR}/include)
set(SOURCE_DIR ${ROOT_DIR}/src)
# set(THIRD_PARTY_DIR ${ROOT_DIR}/third_party) # Si usas dependencias externas

# Incluir directorios de headers
include_directories(
    ${INCLUDE_DIR}
    # ${THIRD_PARTY_DIR}/eigen  # Si usas Eigen desde third_party
    # ${THIRD_PARTY_DIR}/json  # Si incluyes nlohmann/json
)

# Librería principal del proyecto (BERT)
add_library(bert_lib
    ${SOURCE_DIR}/bert.cpp
    # ${SOURCE_DIR}/layers/MultiHeadAttention.cpp
    # ${SOURCE_DIR}/layers/FeedForwardNetwork.cpp
    # ${SOURCE_DIR}/core/EmbeddingLayer.cpp
    # ${SOURCE_DIR}/utils/Tokenizer.cpp
    # Añadir aquí todos los .cpp
)

# =========================== Ejecutable principal ===========================
add_executable(bert_main
    ${ROOT_DIR}/examples/main.cpp  
    # ${ROOT_DIR}/examples/print.cpp
)
# Enlazar dependencias
target_link_libraries(bert_main
    PRIVATE
    bert_lib
    # pthread  # Si se requiere para threading
)


# =========================== Ejecutable de pruebas ===========================
add_executable(bert_test
    # ${ROOT_DIR}/tests/test_vector.cpp
    # ${ROOT_DIR}/tests/test_layers.cpp
    ${ROOT_DIR}/tests/test_embedding.cpp
    # ${ROOT_DIR}/tests/test_tokenizer.cpp
    # ${ROOT_DIR}/tests/test_bert.cpp
    # ${ROOT_DIR}/tests/test_utils.cpp
)
# Enlazar dependencias
target_link_libraries(bert_test
    PRIVATE
    bert_lib
    # pthread  # Si se requiere para threading
)



# Configuración de tests (requiere Google Test u otro framework)
# if(EXISTS ${ROOT_DIR}/tests/CMakeLists.txt)
#     enable_testing()
#     add_subdirectory(${ROOT_DIR}/tests)
# endif()

# Opcional: Configuración para CUDA (si se usa GPU)
# enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES "native")  # Arquitectura específica
# target_compile_options(bert_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)
#include "Tensor.hpp"
#include "Layer.hpp"

class Dense : public Layer { // Clase Dense hereda de Layer
public:
    size_t input_dim;     // Dimension de entrada
    size_t output_dim;    // Dimension de salida
    Tensor weights;       // Pesos de la capa
    Tensor bias;          // Sesgos de la capa
    string activation;    // Tipo de funcion de activacion
    float lambda;         // Coeficiente de weight decay

    Tensor last_input;      // Ultima entrada (x)
    Tensor last_output;     // Ultima salida sin activacion (z)
    Tensor last_activated;  // Ultima salida con activacion (a)
    Tensor grad_weights;    // Gradiente de los pesos
    Tensor grad_bias;       // Gradiente de los sesgos

protected:
    float activation_function(float x) const { // Aplica funcion de activacion
        if (activation == "relu")
            return std::max(0.0f, x);     // ReLU
        else if (activation == "sigmoid")
            return 1.0f / (1.0f + exp(-x)); // Sigmoide
        else if (activation == "tanh")
            return tanh(x);             // Tanh
        else
            return x;  // Identidad
    }

    float activation_derivative(float x) const { // Aplica derivada de activacion
        if (activation == "relu")
            return x > 0 ? 1.0f : 0.0f; // Derivada ReLU
        else if (activation == "sigmoid") {
            float sig = 1.0f / (1.0f + exp(-x));
            return sig * (1.0f - sig);       // Derivada Sigmoide
        } else if (activation == "tanh") {
            float t = tanh(x);
            return 1.0f - t * t;           // Derivada Tanh
        } else
            return 1.0f; // Derivada Identidad
    }

    Tensor softmax(const Tensor& input) {
        const auto& x = input.get_data();
        float max_val = *std::max_element(x.begin(), x.end());

        std::vector<float> exp_values(x.size());
        float sum_exp = 0.0f;
        #pragma omp parallel for reduction(+:sum_exp) // comentar
        for (size_t i = 0; i < x.size(); ++i) {
            exp_values[i] = std::exp(x[i] - max_val); // para evitar overflow
            sum_exp += exp_values[i];
        }

        Tensor result({x.size()});
        auto& r = result.get_data();
        #pragma omp parallel for // comentar
        for (size_t i = 0; i < x.size(); ++i) {
            r[i] = exp_values[i] / sum_exp;
        }
        return result;
    }

public:
    Dense(size_t input_dim_, size_t output_dim_, const std::string& activation_ = "", float lambda_ = 0.0f) { 
        this->input_dim = input_dim_;     // Inicializa dimension entrada
        this->output_dim = output_dim_;   // Inicializa dimension salida
        this->activation = activation_;   // Inicializa activacion
        this->lambda = lambda_; // Inicializa Coeficiente de weight decay
        this->weights = Tensor({input_dim_, output_dim_}); // Crea tensor de pesos
        this->bias = Tensor({output_dim_});             // Crea tensor de sesgos
        this->grad_weights = Tensor({input_dim_, output_dim_}); // Crea tensor gradiente pesos
        this->grad_bias = Tensor({output_dim_});             // Crea tensor gradiente sesgos
        this->initialize_weights_random_uniform(0.1f);
        // if (activation == "relu")
        //     this->initialize_weights_he(input_dim);
        // else if (activation == "sigmoid" || activation == "tanh")
        //     this->initialize_weights_xavier(input_dim, output_dim);
    }

    void initialize_weights_xavier(int input_dim, int output_dim) {
        std::default_random_engine rng; // Generador aleatorio
        float limit = sqrt(6.0f / (input_dim + output_dim)); // Limite distribucion
        std::uniform_real_distribution<float> dist(-limit, limit); // Distribucion uniforme

        // Asigna valores aleatorios
        for (float& w : const_cast<vector<float>&>(weights.get_data())) // pesos
            w = dist(rng); 
        for (float& b : const_cast<vector<float>&>(bias.get_data())) // sesgos
            b = dist(rng); 
    }

    void initialize_weights_he(int input_dim) {
        std::default_random_engine rng; // Generador aleatorio
        float stddev = sqrt(2.0f / input_dim); // Desviacion estandar
        std::normal_distribution<float> dist(0.0f, stddev); // Distribucion normal

        // Asigna valores aleatorios
        for (float& w : const_cast<vector<float>&>(weights.get_data())) // Itera pesos
            w = dist(rng);
        for (float& b : const_cast<vector<float>&>(bias.get_data())) // Itera sesgos
            b = dist(rng);
    }

    void initialize_weights_random_uniform(float range_limit) {
        std::default_random_engine rng;
        // Distribucion uniforme entre -range_limit y +range_limit
        std::uniform_real_distribution<float> dist(-range_limit, range_limit);

        for (float& w : const_cast<vector<float>&>(weights.get_data()))
            w = dist(rng);
        // Los sesgos a menudo se inicializan a cero o a un valor pequenho constante
        for (float& b : const_cast<vector<float>&>(bias.get_data()))
            b = 0.0f; // o se puede usar: dist(rng); 
    }

    const Tensor& get_weights() const { return weights; }
    float get_lambda() const { return lambda; }

    float compute_l2_penalty() const {
        if (lambda == 0.0f) return 0.0f;
        float sum = 0.0f;
        for (float w : weights.get_data()) {
            sum += w * w;
        }
        return lambda * sum;
    }

    void zero_grad() override {
        grad_weights.fill(0.0f);
        grad_bias.fill(0.0f);
    }

    void scale_gradients(float scale) {
        for (float& g : grad_weights.get_data()) g *= scale;
        for (float& g : grad_bias.get_data()) g *= scale;
    }

    Tensor forward(const Tensor& input) override {
        last_input = input;

        // 1. Producto punto input (X) • weights (W)
        last_output = dot_product(input, weights); // Z = X * W

        if (last_activated.get_shape() != std::vector<size_t>{output_dim}) {
            last_activated = Tensor({output_dim});
        }

        auto& z_data = last_output.get_data();  // z sin activacion
        auto& b_data = bias.get_data();         // vector de bias
        auto& a_data = last_activated.get_data(); // salida activada (a)

        // 2. Sumar bias
        #pragma omp parallel for // Se puede comentar esta linea
        for (size_t i = 0; i < output_dim; ++i) {
            z_data[i] += b_data[i];
        }

        // 3. Aplicar función de activacion
        if (activation == "softmax") {
            // Aplicar softmax en todo el vector
            float max_z = *std::max_element(z_data.begin(), z_data.end());
            float sum_exp = 0.0f;
            #pragma omp parallel for reduction(+:sum_exp) // se puede comentar esta linea
            for (size_t i = 0; i < output_dim; ++i)
                sum_exp += std::exp(z_data[i] - max_z);

            #pragma omp parallel for // se puede comentar esta linea
            for (size_t i = 0; i < output_dim; ++i)
                a_data[i] = std::exp(z_data[i] - max_z) / sum_exp;
        } else {
            #pragma omp parallel for // se puede comentar esta linea
            for (size_t i = 0; i < output_dim; ++i)
                a_data[i] = activation_function(z_data[i]);
        }

        return last_activated;
    }

    Tensor backward(const Tensor& grad_output) override {
        const auto& x = last_input.get_data();  // Entrada original (x)
        const auto& z = last_output.get_data(); // Pre-activacion (z = x*w + b)
        const auto& w = weights.get_data();     // Pesos actuales
        auto& gw = grad_weights.get_data();     // Gradiente de pesos
        auto& gb = grad_bias.get_data();        // Gradiente de sesgos

        size_t in = input_dim, out = output_dim;
        const auto& dL_da = grad_output.get_data(); // Gradiente de la perdida respecto a la activacion
        const auto& a = last_activated.get_data(); // a = softmax(z) o activacion normal


        Tensor grad_input({in}); // Inicializa tensor para el gradiente de entrada (dL/dx)
        auto& dL_dx = grad_input.get_data(); // Referencia a los datos de entrada

        if (activation == "softmax") {
            // Asume que la perdida es cross-entropy => grad_output = y_pred - y_true
            for (size_t i = 0; i < out; ++i) {
                float dz = dL_da[i]; // derivada simplificada para softmax + crossentropy
                gb[i] = dz;
                for (size_t j = 0; j < in; ++j) {
                    gw[j * out + i] = dz * x[j];
                    dL_dx[j] += dz * w[j * out + i];
                }
            }
        } else {
            for (size_t i = 0; i < out; ++i) {
                float dz = dL_da[i] * activation_derivative(z[i]);
                gb[i] += dz;
                for (size_t j = 0; j < in; ++j) {
                    gw[j * out + i] += dz * x[j];
                    dL_dx[j] += dz * w[j * out + i];
                }
            }
        }
        
        if (lambda > 0.0f) { // Si hay Coeficiente de weight decay
            auto& gw = grad_weights.get_data();
            const auto& w = weights.get_data();
            for (size_t i = 0; i < w.size(); ++i) {
                gw[i] += 2 * lambda * w[i];
            }
        }

        return grad_input; // Devuelve dL/dx para propagar hacia atrás
    }

    // Nuevo metodo para actualizar parametros usando el optimizador
    void update_parameters(Optimizer& optimizer) override {
        optimizer.update(weights.get_data(), grad_weights.get_data());
        optimizer.update(bias.get_data(), grad_bias.get_data());
    }

    void print() const override { // Imprime informacion de la capa
        cout << "Dense Layer (" << input_dim << " -> " << output_dim << "), activation: " << activation << endl;
        cout << "Weights:\n"; weights.print();
        cout << "Bias:\n"; bias.print();
    }

    void save(std::ostream& out) const { // Guarda la capa en un archivo
        out << input_dim << " " << output_dim << " " << activation << "\n"; // Guarda dimensiones y activacion
        for (float w : weights.get_data()) out << w << " ";  // Guarda pesos
        out << "\n";
        for (float b : bias.get_data()) out << b << " ";  // Guarda sesgos
        out << "\n";
    }

    void load(std::istream& in) { // Carga la capa desde un archivo
        string act; // Variable para la funcion de activacion
        in >> input_dim >> output_dim >> act; // Lee dimensiones y activacion
        activation = act; // Asigna activacion

        weights = Tensor({input_dim, output_dim}); // Recrea tensor pesos
        bias = Tensor({output_dim});               // Recrea tensor sesgos

        for (float& w : weights.get_data()) in >> w;  // Lee pesos
        for (float& b : bias.get_data()) in >> b;     // Lee sesgos
    }
};
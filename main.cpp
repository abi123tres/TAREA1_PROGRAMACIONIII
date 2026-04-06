#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Tensor;

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() {}
};

class Tensor {
private:
    vector<size_t> shape;
    double *data;
public:
    Tensor() {
        data = nullptr;
    }
    friend class TensorTransform;
    friend class ReLU;
    friend class Sigmoid;
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor (const std :: vector < size_t >& shape, const std :: vector < double >& values ) {
        this->shape = shape;
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total_size *= shape[i];
        }
        if (values.size() != total_size) {
            cout << "Los tamanhos no coinciden" << endl;
            data = nullptr;
            return;
        }
        data = new double[total_size];
        for (size_t i = 0; i < total_size; i++) {
            data[i] = values[i];
        }
    }
    static Tensor zeros(const vector<size_t>& shape) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total_size *= shape[i];
        }
        vector<double> values(total_size, 0);
        return Tensor(shape, values);
    }
    static Tensor ones(const vector<size_t>& shape) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total_size *= shape[i];
        }
        vector<double> values(total_size, 1);
        return Tensor(shape, values);
    }
    static Tensor random(const vector<size_t>& shape, double min, double max) {
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total_size *= shape[i];
        }
        vector<double> values;
        for (size_t i = 0; i < total_size; i++) {
            values.push_back(min + (max - min) * rand() / (RAND_MAX + 1.0));
        }
        return Tensor(shape, values);
    }
    static Tensor arange(double start, double end) {
        vector<double> values;
        for (double i = start; i <= end; i++) {
            values.push_back(i);
        }
        vector<size_t> shape = {values.size()};
        return Tensor(shape, values);
    }
    Tensor ( const Tensor & other ) {
        shape = other.shape;
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        data = new double[total];
        for (size_t i = 0; i < total; i++) {
            data[i] = other.data[i];
        }
    }
    Tensor ( Tensor && other ) noexcept {
        shape = other.shape;
        data = other.data;
        other.data = nullptr;
    }
    Tensor & operator =( const Tensor & other ) {
        if (this == &other) return *this;
        delete[] data;
        shape = other.shape;
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        data = new double[total];
        for (size_t i = 0; i < total; i++) {
            data[i] = other.data[i];
        }
        return *this;
    }
    Tensor & operator =( Tensor && other ) noexcept {
        if (this == &other) return *this;
        delete[] data;
        shape = other.shape;
        data = other.data;
        other.data = nullptr;
        return *this;
    }
    ~ Tensor () {
        delete[] data;
    }
    Tensor apply(const TensorTransform& transform) const {
        return transform.apply(*this);
    }
    Tensor operator+(const Tensor& other) const {
        if (shape.size() != other.shape.size()) {
            cout << "Error dimensiones" << endl;
            return Tensor();
        }

        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != other.shape[i] && other.shape[i] != 1) {
                cout << "Error dimensiones" << endl;
                return Tensor();
            }
        }
        size_t total_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total_size *= shape[i];
        }
        vector<double> values(total_size);
        for (size_t i = 0; i < total_size; i++) {
            size_t j = i % other.shape.back();
            values[i] = data[i] + other.data[j];
        }
        return Tensor(shape, values);
    }
    Tensor operator-(const Tensor& other) const {
        if (shape != other.shape) {
            cout << "Los tamanhos no coincide" << endl;
            return Tensor();
        }
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        vector<double> values(total);
        for (size_t i = 0; i < total; i++) {
            values[i] = data[i] - other.data[i];
        }
        return Tensor(shape, values);
    }
    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape) {
            cout << "Los tamanhos no coincide" << endl;
            return Tensor();
        }
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        vector<double> values(total);
        for (size_t i = 0; i < total; i++) {
            values[i] = data[i] * other.data[i];
        }
        return Tensor(shape, values);
    }
    Tensor operator* (double n) const {
        size_t total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        vector<double> values(total);
        for (size_t i = 0; i < total; i++) {
            values[i] = data[i] * n;
        }
        return Tensor(shape, values);
    }
    Tensor view(const vector<size_t>& nuevo_shape) const {
        size_t viejo_size = 1;
        for (size_t i = 0; i < this->shape.size(); i++) {
            viejo_size *= this->shape[i];
        }
        size_t nuevo_size = 1;
        for (size_t i = 0; i < nuevo_shape.size(); i++) {
            nuevo_size *= nuevo_shape[i];
        }
        if (viejo_size != nuevo_size) {
            cout << "Los tamanhos no coinciden" << endl;
            return Tensor();
        }
        vector<double> values(viejo_size);
        for (size_t i = 0; i < viejo_size; i++) {
            values[i] = this->data[i];
        }
        return Tensor(nuevo_shape, values);
    }
    Tensor unsqueeze(size_t posicion) const {
        if (shape.size() > 3) {
            cout << "No se permiten mas de 3 dimensiones" << endl;
            return Tensor();
        }
        if (posicion > shape.size()) {
            cout << "La posicion es invalida" << endl;
            return Tensor();
        }
        vector<size_t> nuevo_shape = shape;
        nuevo_shape.insert(nuevo_shape.begin() + posicion,1);
        Tensor result;
        result.shape = shape;
        return *this;
    }
    static Tensor concat(const vector<Tensor>& tensors, size_t pos) {
    const vector<size_t>& aux = tensors[0].shape;
    for (size_t i = 1; i < tensors.size(); i++) {
        if (tensors[i].shape.size() != aux.size()) {
            cout << "Dimensiones incompatibles" << endl;
            return Tensor();
        }
        for (size_t j = 0; j < aux.size(); j++) {
            if (j != pos && tensors[i].shape[j] != aux[j]) {
                cout << "Las dimensiones no coinciden" << endl;
                return Tensor();
            }
        }
    }
    vector<size_t> nuevo_shape = aux;
    nuevo_shape[pos] = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        nuevo_shape[pos] += tensors[i].shape[pos];
    }
    size_t total_size = 1;
    for (size_t i = 0; i < nuevo_shape.size(); i++) {
        total_size *= nuevo_shape[i];
    }
    double* new_data = new double[total_size];
    size_t idx = 0;
    if (pos == 0) {
        for (size_t i = 0; i < tensors.size(); i++) {
            size_t tensor_size = 1;
            for (size_t j = 0; j < tensors[i].shape.size(); j++) {
                tensor_size *= tensors[i].shape[j];
            }
            for (size_t k = 0; k < tensor_size; k++) {
                new_data[idx] = tensors[i].data[k];
                idx++;
            }
        }
    }
    else if (pos == 1) {
        size_t filas = aux[0];
        for (size_t i = 0; i < filas; i++) {
            for (size_t j = 0; j < tensors.size(); j++) {
                size_t col = tensors[j].shape[1];
                for (size_t k = 0; k < col; k++) {
                    new_data[idx] = tensors[j].data[i * col + k];
                    idx++;
                }
            }
        }
    }
    else if (pos == 2) {
        size_t d1 = aux[0];
        size_t d2 = aux[1];
        for (size_t i = 0; i < d1; i++) {
            for (size_t j = 0; j < d2; j++) {
                for (size_t k = 0; k < tensors.size(); k++) {
                    size_t profundidad = tensors[k].shape[2];
                    for (size_t l = 0; l < profundidad; l++) {
                        size_t ind = i * (d2 * profundidad) + j * profundidad + l;
                        new_data[idx] = tensors[k].data[ind];
                        idx++;
                    }
                }
            }
        }
    }
    Tensor result;
    result.shape = nuevo_shape;
    result.data = new_data;
    return move(result);
    }
};

Tensor dot(const Tensor& a, const Tensor& b) {
    size_t size_a = 1;
    for (size_t i = 0; i < a.shape.size(); i++) {
        size_a *= a.shape[i];
    }
    size_t size_b = 1;
    for (size_t i = 0; i < b.shape.size(); i++) {
        size_b *= b.shape[i];
    }
    if (size_a != size_b) {
        cout << "Los tamanhos no coinciden" << endl;
        return Tensor();
    }
    double resultado = 0;
    for (size_t i = 0; i < size_a; i++) {
        resultado += a.data[i] * b.data[i];
    }
    vector<size_t> tensor_result = {1};
    vector<double> values = {resultado};
    Tensor resultad (tensor_result,values);
    return move(resultad);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        cout << "No son vectors bidimensionales" << endl;
        return Tensor();
    }
    size_t x = a.shape[0];
    size_t y = a.shape[1];
    size_t z = b.shape[1];
    if (y != b.shape[0] ) {
        cout << "Dimensiones incompatibles" << endl;
        return Tensor();
    }
    vector<size_t> nuevo_shape = {x,z};
    vector<double> values(x * z, 0.0);
    for (size_t i = 0; i < x; i++) {
        for (size_t j = 0; j < z; j++) {
            double suma = 0;
            for (size_t k = 0; k < y; k++) {
                suma += a.data[i * y + k] * b.data[k * z + j];
            }
            values[i * z + j] = suma;
        }
    }
    Tensor result(nuevo_shape, values);
    return move(result);
}

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        size_t total = 1;
        for (size_t i = 0; i < t.shape.size(); i++) {
            total *= t.shape[i];
        }
        vector<double> values(total);
        for (size_t i = 0; i < total; i++) {
            values[i] = (t.data[i] > 0) ? t.data[i] : 0;
        }
        return Tensor(t.shape, values);
    }
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        size_t total = 1;
        for (size_t i = 0; i < t.shape.size(); i++) {
            total *= t.shape[i];
        }

        vector<double> values(total);

        for (size_t i = 0; i < total; i++) {
            values[i] = 1.0 / (1.0 + exp(-t.data[i]));
        }

        return Tensor(t.shape, values);
    }
};


int main() {
    Tensor input({1000,20,20}, vector<double>(1000*20*20, 1.0));
    Tensor x = input.view({1000,400});
    Tensor W1({400,100}, vector<double>(400*100, 0.5));
    Tensor z1 = matmul(x, W1);
    Tensor b1({1,100}, vector<double>(100, 0.1));
    Tensor z1_bias = z1 + b1;
    ReLU relu;
    Tensor a1 = z1_bias.apply(relu);
    Tensor W2({100,10}, vector<double>(100*10, 0.3));
    Tensor z2 = matmul(a1, W2);
    Tensor b2({1,10}, vector<double>(10, 0.2));
    Tensor z2_bias = z2 + b2;
    Sigmoid sigmoid;
    Tensor output = z2_bias.apply(sigmoid);
    return 0;
}
#include "../include/matrix.hpp"

// Constructor
Matrix::Matrix() : rows(0), cols(0), data(nullptr) { }

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data = new double[rows * cols];
}

Matrix::Matrix(int rows, int cols, double val) : rows(rows), cols(cols) {
    data = new double[rows * cols];
    for (int i = 0; i < rows* cols; i++) {
        data[i] = val;
    }
}

// Destructor
Matrix::~Matrix() {
    delete[] data;
}

double Matrix::get(int offset) const {
    return data[offset];
}

// Operator overloading
double* Matrix::operator[](int row) {
    return &data[row * cols];  // Returns pointer to the row start
}

const double* Matrix::operator[](int row) const {
    return &data[row * cols];  // Const version
}


Matrix Matrix::operator+(const Matrix& other) const{
    return add(other);
}
Matrix Matrix::operator-(const Matrix& other) const {
    return subtract(other);
}
Matrix Matrix::operator*(double scalar) const{
    return multiply(scalar);
}
Matrix Matrix::operator*(const Matrix& other) const{
    return multiply(other);
}
Matrix Matrix::operator/(double scalar) const{
    return multiply(1.0 / scalar);
}
Matrix& Matrix::operator+=(const Matrix& other){
    *this = add(other);
    return *this;
}
Matrix& Matrix::operator-=(const Matrix& other){
    *this = subtract(other);
    return *this;
}
Matrix& Matrix::operator*=(double scalar){
    *this = multiply(scalar);
    return *this;
}
Matrix& Matrix::operator*=(const Matrix& other){
    *this = multiply(other);
    return *this;
}
Matrix& Matrix::operator/=(double scalar){
    *this = multiply(1.0 / scalar);
    return *this;
}
bool Matrix::operator==(const Matrix& other) const{
    for(int i = 0; i < rows * cols; i++){
        if(data[i] != other.data[i]){
            return false;
        }
    }
    return true;
}
bool Matrix::operator!=(const Matrix& other) const{
    return !(*this == other);
}


// Getters
double Matrix::getRows() const { return rows; }
double Matrix::getCols() const { return cols; }
double Matrix::get(int i, int j) const { return data[i * cols + j]; }


// Print function
void Matrix::print() const {
    int counter = 0;
    int counter2 = 0;
    for (int i = 0; i < rows* cols; i++) {
        std::cout << data[counter] << " ";
        counter++;
        counter2++;
        if (counter2 == cols) {
            std::cout << std::endl;
            counter2 = 0;
        }
    }
}

// Add two matrices
Matrix Matrix::add(const Matrix& other) const {
    Matrix result(rows, cols, 0.0);
    for (int i = 0; i < rows* cols; i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// Subtract two matrices
Matrix Matrix::subtract(const Matrix& other) const {
    Matrix result(rows, cols, 0.0);
    for (int i = 0; i < rows* cols; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Multiply matrix by scalar
Matrix Matrix::multiply(double scalar) const {
    Matrix result(rows, cols, 0.0);
    for (int i = 0; i < rows* cols; i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Multiply two matrices
Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    }

    Matrix result(rows, other.cols, 0.0);

    // Multiply matrices
    //todo could be optimized
    for (int i = 0; i < rows; i++) {
        int row_offset = i * cols;
        int other_col_offset = i* other.cols;
        for (int k = 0; k < cols; k++) {  // Move k loop outwards
            int k_offset = k * other.cols;
            int temp = data[row_offset + k];  // Cache data[i][k]
            for (int j = 0; j < other.cols; j++) {
                result.data[other_col_offset + j] += temp * other.data[k_offset + j];
            }
        }
    }
    return result;
}

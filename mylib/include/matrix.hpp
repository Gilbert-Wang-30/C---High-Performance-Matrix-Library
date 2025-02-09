#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>

class Matrix {
private:
    double* data;
    double* data_T;
    int rows, cols;

    double get(int offset) const;

public:
    // Constructor
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double val);

    // Destructor
    ~Matrix();

    // Operator overloading
    double get_row(int i, int j) const;
    double get_col(int i, int j) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator/(double scalar) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix &operator*=(double scalar);
    Matrix &operator*=(int scalar);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator/=(double scalar);
    inline bool operator==(const Matrix& other) const;
    inline bool operator!=(const Matrix& other) const;

    // Setter functions
    double set(int i, int j, double val);

    // Getter functions
    double getRows() const;
    double getCols() const;

    // Basic operations
    void print() const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(double scalar) const;
    Matrix multiply(const Matrix& other) const;
};

#endif // MATRIX_HPP

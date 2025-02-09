#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>

class Matrix {
private:
    double* data;
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
    double get(int i, int j) const;
    double* operator[](int row);
    const double* operator[](int row) const;
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

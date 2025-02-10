#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <immintrin.h>  // AVX, AVX2 intrinsics



inline double _mm256_reduce_add_pd(__m256d v) {
    __m128d sum1 = _mm256_castpd256_pd128(v);
    __m128d sum2 = _mm256_extractf128_pd(v, 1);
    sum1 = _mm_add_pd(sum1, sum2);
    return _mm_cvtsd_f64(_mm_hadd_pd(sum1, sum1));
}




class Matrix {
private:
    int rows, cols;
    double* data;
    double* data_T;
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

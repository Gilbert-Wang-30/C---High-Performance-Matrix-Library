#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <immintrin.h>  // AVX, AVX2 intrinsics

inline double _mm256_reduce_add_pd(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d vsum  = _mm_add_pd(vlow, vhigh);
    return _mm_cvtsd_f64(_mm_hadd_pd(vsum, vsum));
}


// inline double _mm512_reduce_add_pd(__m512d v) {
//     __m256d low = _mm512_castpd512_pd256(v);
//     __m256d high = _mm512_extractf64x4_pd(v, 1);
//     __m256d sum256 = _mm256_add_pd(low, high);
//     __m128d low128 = _mm256_castpd256_pd128(sum256);
//     __m128d high128 = _mm256_extractf128_pd(sum256, 1);
//     __m128d sum128 = _mm_add_pd(low128, high128);
//     return _mm_cvtsd_f64(_mm_hadd_pd(sum128, sum128));
// }



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

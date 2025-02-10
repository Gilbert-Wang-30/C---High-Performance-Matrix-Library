#include "../include/matrix.hpp"
#include "../include/simd_utils.hpp"
#include <immintrin.h>  // AVX, AVX2 intrinsics
#ifdef __ARM_NEON
    #include <arm_neon.h>  // Only include NEON if compiling for macOS ARM
#endif



// Constructor
Matrix::Matrix() : rows(0), cols(0), data(nullptr), data_T(nullptr) { }

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data = new double[rows * cols];
    data_T = new double[rows * cols];
}

Matrix::Matrix(int rows, int cols, double val) : rows(rows), cols(cols) {
    data = new double[rows * cols];
    data_T = new double[rows * cols];
    for (int i = 0; i < rows* cols; i++) {
        data[i] = val;
        data_T[i] = val;
    }
}

// Destructor
Matrix::~Matrix() {
    delete[] data;
    delete[] data_T;
}

double Matrix::get(int offset) const {
    return data[offset];
}

double Matrix::get_row(int i, int j) const {
    return data[i * cols + j];
}

double Matrix::get_col(int i, int j) const {
    return data_T[j * rows + i];
}

// Operator overloading
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

// Setter function
double Matrix::set(int i, int j, double val) {
    data[i * cols + j] = val;
    data_T[j * rows + i] = val;
    return val;
}

// Getters
double Matrix::getRows() const { return rows; }
double Matrix::getCols() const { return cols; }


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
        result.data_T[i] = data_T[i] + other.data_T[i];
    }
    return result;
}

// Subtract two matrices
Matrix Matrix::subtract(const Matrix& other) const {
    Matrix result(rows, cols, 0.0);
    for (int i = 0; i < rows* cols; i++) {
        result.data[i] = data[i] - other.data[i];
        result.data_T[i] = data_T[i] - other.data_T[i];
    }
    return result;
}

// Multiply matrix by scalar
Matrix Matrix::multiply(double scalar) const {
    Matrix result(rows, cols, 0.0);
    for (int i = 0; i < rows* cols; i++) {
        result.data[i] = data[i] * scalar;
        result.data_T[i] = data_T[i] * scalar;
    }
    return result;
}

// Multiply two matrices
Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    }

    Matrix result(rows, other.cols, 0.0);
    const int BLOCK_SIZE = 32;  // Tune for performance
    

    if (hasAVX512()) {
        #ifdef __AVX512F__
        std::cout << "Using AVX-512 optimization\n";
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            for (int j = 0; j < other.cols; j += BLOCK_SIZE) {
                for (int k = 0; k < cols; k += BLOCK_SIZE) {
                    for (int ii = i; ii < std::min(i + BLOCK_SIZE, rows); ii++) {
                        int ii_offset = ii * cols;
                        int result_offset = ii * other.cols;

                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, other.cols); jj++) {
                            int jj_offset = jj * other.rows;
                            double sum = 0.0;

                            for (int kk = k; kk < std::min(k + BLOCK_SIZE, cols); kk += 8) {
                                __m512d a = _mm512_loadu_pd(&data[ii_offset + kk]);
                                __m512d b = _mm512_loadu_pd(&other.data_T[jj_offset + kk]);
                                __m512d c = _mm512_mul_pd(a, b);
                                sum += _mm512_reduce_add_pd(c);  //
                            }

                            result.data[result_offset + jj] += sum;
                        }
                    }
                }
            }
        }
        #endif
    } else if (hasAVX2()) {
        #ifdef __AVX2__
        std::cout << "Using AVX2 optimization\n";
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            for (int j = 0; j < other.cols; j += BLOCK_SIZE) {
                for (int k = 0; k < cols; k += BLOCK_SIZE) {
                    for (int ii = i; ii < std::min(i + BLOCK_SIZE, rows); ii++) {
                        int ii_offset = ii * cols;
                        int result_offset = ii * other.cols;

                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, other.cols); jj++) {
                            int jj_offset = jj * other.rows;
                            double sum = 0.0;

                            for (int kk = k; kk < std::min(k + BLOCK_SIZE, cols); kk += 4) {
                                __m256d a = _mm256_loadu_pd(&data[ii_offset + kk]);
                                __m256d b = _mm256_loadu_pd(&other.data_T[jj_offset + kk]);
                                __m256d c = _mm256_mul_pd(a, b);
                                sum += _mm256_reduce_add_pd(c);
                            }

                            result.data[result_offset + jj] += sum;
                        }
                    }
                }
            }
        }
        #endif
    }

    #ifdef __ARM_NEON  // Compile NEON code ONLY on macOS (ARM)
    else if (hasNEON()) {
        std::cout << "Using NEON optimization\n";
        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            for (int j = 0; j < other.cols; j += BLOCK_SIZE) {
                for (int k = 0; k < cols; k += BLOCK_SIZE) {
                    for (int ii = i; ii < std::min(i + BLOCK_SIZE, rows); ii++) {
                        int ii_offset = ii * cols;
                        int result_offset = ii * other.cols;

                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, other.cols); jj++) {
                            int jj_offset = jj * other.rows;
                            double sum = 0.0;

                            for (int kk = k; kk < std::min(k + BLOCK_SIZE, cols); kk += 2) {
                                float64x2_t a = vld1q_f64(&data[ii_offset + kk]);
                                float64x2_t b = vld1q_f64(&other.data_T[jj_offset + kk]);
                                float64x2_t c = vmulq_f64(a, b);
                                sum += vaddvq_f64(c);
                            }

                            result.data[result_offset + jj] += sum;
                        }
                    }
                }
            }
        }
    }
    #endif  // __ARM_NEON
    else {
        std::cout << "Using scalar fallback\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i * cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
    }

    return result;
}

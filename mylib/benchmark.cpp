#include <iostream>
#include <chrono>
#include "../include/matrix.hpp"

using namespace std;
using namespace std::chrono;

void benchmarkMatrixMultiplication(int size) {
    Matrix A(size, size, 1.0);  // Initialize matrices with all ones
    Matrix B(size, size, 2.0);

    cout << "Benchmarking " << size << "x" << size << " matrix multiplication...\n";

    auto start = high_resolution_clock::now();
    Matrix C = A * B;  // Perform optimized multiplication
    auto end = high_resolution_clock::now();

    double elapsed = duration<double>(end - start).count();
    cout << "Execution time: " << elapsed << " seconds\n\n";
}

int main() {
    benchmarkMatrixMultiplication(128);
    benchmarkMatrixMultiplication(256);
    benchmarkMatrixMultiplication(512);
    benchmarkMatrixMultiplication(1024);  // Large test case (if system can handle it)
    benchmarkMatrixMultiplication(2048);  // Large test case (if system can handle it)
    benchmarkMatrixMultiplication(4096);  // Large test case (if system can handle it)
    //benchmarkMatrixMultiplication(8192);  // Large test case (if system can handle it)
    //benchmarkMatrixMultiplication(16384);  // Large test case (if system can handle it)

    return 0;
}

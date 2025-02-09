#include <iostream>
#include <chrono>
#include "include/matrix.hpp"

void benchmarkMultiplication(int size, bool print = false) {
    // Create random matrices
    Matrix A(size, size, 1.0);
    Matrix B(size, size, 2.0);

    // Time the multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = A * B;  // Optimized multiplication
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Matrix Multiplication (" << size << "x" << size << "): " 
              << elapsed << " seconds\n";

    if (print) {
        C.print();
    }
}

int main() {
    std::cout << "Benchmarking Matrix Multiplication...\n";
    
    benchmarkMultiplication(128);  // Small test
    benchmarkMultiplication(256);  // Medium test
    benchmarkMultiplication(512);  // Large test
    benchmarkMultiplication(1024); // Very large test

    return 0;
}

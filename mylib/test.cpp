#include <iostream>
#include "include/matrix.hpp"
#include <cassert>

using namespace std;

// Function to compare two matrices with a small tolerance for floating-point precision
bool areMatricesEqual(const Matrix& A, const Matrix& B, double tol = 1e-6) {
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) return false;
    for (int i = 0; i < A.getRows(); i++) {
        for (int j = 0; j < A.getCols(); j++) {
            if (std::abs(A.get_row(i, j) - B.get_row(i, j)) > tol) return false;
        }
    }
    return true;
}


void runBasicArethmeticTests() {
    Matrix A(11, 13, 0.0);
    Matrix B(13, 15, 0.0);

    // Set specific values for Matrix A
    A.set(0, 1, 23.45);
    A.set(1, 5, 12.78);
    A.set(2, 9, 34.56);
    A.set(3, 2, 78.91);
    A.set(4, 6, 45.32);
    A.set(5, 10, 67.89);
    A.set(6, 3, 89.01);
    A.set(7, 7, 56.78);
    A.set(8, 12, 90.12);
    A.set(9, 0, 34.98);
    A.set(10, 11, 78.23);

    // Set specific values for Matrix B
    B.set(0, 2, 11.34);
    B.set(1, 4, 22.56);
    B.set(2, 6, 33.78);
    B.set(3, 8, 44.90);
    B.set(4, 10, 55.12);
    B.set(5, 12, 66.34);
    B.set(6, 14, 77.56);
    B.set(7, 1, 88.78);
    B.set(8, 3, 99.90);
    B.set(9, 5, 10.12);
    B.set(10, 7, 21.34);
    B.set(11, 9, 32.56);
    B.set(12, 11, 43.78);

    // Perform multiplication
    Matrix C = A * B;
    // Matrix A(2, 2, 1.0);
    // Matrix B(2, 2, 2.0);
    //Matrix C = A * B;

    cout << "Test 1: Basic 2x2 Multiplication\n";
    C.print();
    double x = C.get_row(1, 12);
    cout << "Value at row 1, col 12: " << x << endl;
    Matrix Zero(2, 2, 0.0);
    A = Matrix(2, 2, 1.0);
    Matrix D = A * Zero;

    cout << "Test 2: Multiplication by Zero Matrix\n";
    D.print();

    Matrix Identity(2, 2);
    Identity.set(0, 0, 1.0);
    Identity.set(1, 1, 1.0); 

    Matrix E = A * Identity;

    cout << "Test 3: Multiplication by Identity Matrix\n";
    E.print();

    Matrix F(3, 2, 1.0);
    Matrix G(2, 3, 2.0);
    Matrix H = F * G;

    cout << "Test 4: Rectangular Matrices (3x2 * 2x3)\n";
    H.print();

    Matrix I(3, 4, 1.0);
    Matrix J(4, 2, 2.0);
    Matrix K = I * J;
    cout << "K: " << K.getRows() << "x" << K.getCols() << endl;

    cout << "Test 5: Non-Square Multiplication (3x4 * 4x2)\n";
    K.print();

    Matrix Large1(100, 100, 1.0);
    Matrix Large2(100, 100, 1.0);
    Matrix LargeResult = Large1 * Large2;

    cout << "Test 6: Large 100x100 Multiplication Completed\n";

    try {
        Matrix L(2, 3, 1.0);
        Matrix M(4, 2, 1.0);
        Matrix N = L * M;  // Should throw an error!
        cout << "Test 7 Failed: No error thrown!\n";
    } catch (const std::invalid_argument& e) {
        cout << "Test 7 Passed: Caught exception - " << e.what() << "\n";
    }

    Matrix Sparse1(3, 3, 0.0);
    Sparse1.set(0, 1, 5);
    Sparse1.set(2, 2, 10);

    Matrix Sparse2(3, 3, 1.233434334);
    Sparse2.set(1, 0, 2);
    Sparse2.set(2, 1, 4);

    Matrix SparseResult = Sparse1 * Sparse2;

    cout << "Test 8: Sparse Matrix Multiplication\n";
    SparseResult.print();

    // Large matrix multiplication with known result
    int size = 10;
    Matrix M1(size, size, 1.0);
    Matrix M2(size, size, 2.0);
    Matrix M3 = M1 * M2;

    cout << "Test 9: Large Matrix Multiplication (10x10) with known result\n";
    M3.print();
}


// ✅ 1️⃣ Arithmetic Operations Test
void runArithmeticTests() {
    std::cout << "Running Arithmetic Tests..." << std::endl;

    Matrix A(2, 2, 3.0);  // 2x2 matrix filled with 3.0
    Matrix B(2, 2, 1.5);  // 2x2 matrix filled with 1.5

    // Addition
    Matrix C = A + B;
    assert(areMatricesEqual(C, Matrix(2, 2, 4.5)));

    // Subtraction
    C = A - B;
    assert(areMatricesEqual(C, Matrix(2, 2, 1.5)));

    // Scalar Multiplication
    C = A * 2.0;
    assert(areMatricesEqual(C, Matrix(2, 2, 6.0)));

    // Scalar Division
    C = A / 3.0;
    assert(areMatricesEqual(C, Matrix(2, 2, 1.0)));

    std::cout << "✅ Arithmetic Tests Passed!" << std::endl;
}

// ✅ 2️⃣ Move Assignment Test
void runMoveAssignmentTests() {
    std::cout << "Running Move Assignment Tests..." << std::endl;

    Matrix A(3, 3, 5.0);
    Matrix B = std::move(A); // Move constructor

    assert(A.getRows() == 0 && A.getCols() == 0);  // A should be in a null state
    assert(areMatricesEqual(B, Matrix(3, 3, 5.0)));

    Matrix C(3, 3, 2.0);
    C = std::move(B); // Move assignment

    assert(B.getRows() == 0 && B.getCols() == 0);  // B should be in a null state
    assert(areMatricesEqual(C, Matrix(3, 3, 5.0)));

    std::cout << "✅ Move Assignment Tests Passed!" << std::endl;
}

// ✅ 3️⃣ Matrix Multiplication Tests
void runMultiplicationTests() {
    std::cout << "Running Multiplication Tests..." << std::endl;

    Matrix A(2, 2, 1.0);
    Matrix B(2, 2, 2.0);
    Matrix C = A * B;  // Matrix multiplication

    assert(areMatricesEqual(C, Matrix(2, 2, 4.0))); // 1x2 + 1x2 for each element

    std::cout << "✅ Multiplication Tests Passed!" << std::endl;
}

// ✅ 4️⃣ Comparison Operators Tests
void runComparisonTests() {
    std::cout << "Running Comparison Tests..." << std::endl;

    Matrix A(2, 2, 3.0);
    Matrix B(2, 2, 3.0);
    Matrix C(2, 2, 4.0);

    assert(A == B);
    assert(A != C);

    std::cout << "✅ Comparison Tests Passed!" << std::endl;
}

// ✅ 5️⃣ Transpose Tests
void runTransposeTests() {
    std::cout << "Running Transpose Tests..." << std::endl;

    Matrix A(2, 2);
    A.set(0, 0, 1);
    A.set(0, 1, 2);
    A.set(1, 0, 3);
    A.set(1, 1, 4);

    Matrix T = A.get_transpose(); // Assuming multiply handles transpose

    assert(T.get_row(0, 1) == 3);
    assert(T.get_row(1, 0) == 2);

    std::cout << "✅ Transpose Tests Passed!" << std::endl;
}

// ✅ 6️⃣ Edge Cases: Empty Matrix, Zero Matrix, Large Matrix
void runEdgeCaseTests() {
    std::cout << "Running Edge Case Tests..." << std::endl;

    Matrix emptyMatrix;
    assert(emptyMatrix.getRows() == 0 && emptyMatrix.getCols() == 0);

    Matrix zeroMatrix(3, 3, 0.0);
    assert(areMatricesEqual(zeroMatrix, Matrix(3, 3, 0.0)));

    std::cout << "✅ Edge Case Tests Passed!" << std::endl;
}

// ✅ 7️⃣ In-Place Operations Tests (+=, -=, *=)
void runInPlaceOperationsTests() {
    std::cout << "Running In-Place Operations Tests..." << std::endl;

    Matrix A(2, 2, 2.0);
    Matrix B(2, 2, 3.0);

    A += B;
    assert(areMatricesEqual(A, Matrix(2, 2, 5.0)));

    A -= B;
    assert(areMatricesEqual(A, Matrix(2, 2, 2.0)));

    A *= 2;
    assert(areMatricesEqual(A, Matrix(2, 2, 4.0)));

    std::cout << "✅ In-Place Operations Tests Passed!" << std::endl;
}

// ✅ 8️⃣ Large Matrix Performance Test
void runLargeMatrixTest() {
    std::cout << "Running Large Matrix Test..." << std::endl;

    Matrix A(1000, 1000, 1.0);
    Matrix B(1000, 1000, 2.0);

    Matrix C = A + B;
    assert(C.get_row(500, 500) == 3.0); // Mid-point check

    std::cout << "✅ Large Matrix Test Passed!" << std::endl;
}

// ✅ 9️⃣ Exception Handling Tests (if applicable)
void runExceptionTests() {
    std::cout << "Running Exception Handling Tests..." << std::endl;

    try {
        Matrix A(2, 2);
        Matrix B(3, 3);
        Matrix C = A * B; // Should throw exception due to size mismatch
        assert(false); // Should not reach here
    } catch (...) {
        std::cout << "✅ Exception Handling Test Passed!" << std::endl;
    }
}

// ✅ 🔟 Final Run Function
void runAllTests() {
    runArithmeticTests();
    runMoveAssignmentTests();
    runMultiplicationTests();
    runComparisonTests();
    runTransposeTests();
    runEdgeCaseTests();
    runInPlaceOperationsTests();
    runLargeMatrixTest();
    runExceptionTests();
}

// ✅ Main Function to Run All Tests
int main() {
    runAllTests();
    std::cout << "🎉 ALL TESTS PASSED! 🎉" << std::endl;
    return 0;
}


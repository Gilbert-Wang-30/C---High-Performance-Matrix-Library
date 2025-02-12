#include <iostream>
#include "include/matrix.hpp"

using namespace std;



void runArethmeticTests() {
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


void runOperatorTests(){
    
}


int main() {
    //runArethmeticTests();
    return 0;
}

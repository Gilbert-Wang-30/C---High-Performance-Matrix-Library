#include <iostream>
#include "include/matrix.hpp"

using namespace std;

void runTests() {
    Matrix A(2, 2, 1.0);
    Matrix B(2, 2, 2.0);
    Matrix C = A * B;

    cout << "✅ Test 1: Basic 2x2 Multiplication\n";
    C.print();

    Matrix Zero(2, 2, 0.0);
    Matrix D = A * Zero;

    cout << "✅ Test 2: Multiplication by Zero Matrix\n";
    D.print();

    Matrix Identity(2, 2);
    Identity[0][0] = 1.0; Identity[1][1] = 1.0;  // Manually set diagonal to 1

    Matrix E = A * Identity;

    cout << "✅ Test 3: Multiplication by Identity Matrix\n";
    E.print();

    Matrix F(3, 2, 1.0);
    Matrix G(2, 3, 2.0);
    Matrix H = F * G;

    cout << "✅ Test 4: Rectangular Matrices (3x2 * 2x3)\n";
    H.print();

    Matrix I(3, 4, 1.0);
    Matrix J(4, 2, 2.0);
    Matrix K = I * J;
    cout << "K: " << K.getRows() << "x" << K.getCols() << endl;

    cout << "✅ Test 5: Non-Square Multiplication (3x4 * 4x2)\n";
    K.print();

    Matrix Large1(100, 100, 1.0);
    Matrix Large2(100, 100, 1.0);
    Matrix LargeResult = Large1 * Large2;

    cout << "✅ Test 6: Large 100x100 Multiplication Completed\n";

    try {
        Matrix L(2, 3, 1.0);
        Matrix M(4, 2, 1.0);
        Matrix N = L * M;  // ❌ Should throw an error!
        cout << "❌ Test 7 Failed: No error thrown!\n";
    } catch (const std::invalid_argument& e) {
        cout << "✅ Test 7 Passed: Caught exception - " << e.what() << "\n";
    }
    Matrix Sparse1(3, 3, 0.0);
    Sparse1[0][1] = 5;
    Sparse1[2][2] = 10;

    Matrix Sparse2(3, 3, 1.233434334);
    Sparse2[1][0] = 2;
    Sparse2[2][1] = 4;

    Matrix SparseResult = Sparse1 * Sparse2;

    cout << "✅ Test 8: Sparse Matrix Multiplication\n";
    SparseResult.print();
    

}

int main() {
    runTests();
    return 0;
}

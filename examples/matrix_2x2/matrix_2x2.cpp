#include "mat.hpp"

using namespace cv;

int main(int argc, char **argv)
{
  std::vector<uint32_t> data1{1, 2, 3, 4};
  std::vector<uint32_t> data2{5, 6, 7, 8};

  auto size = Size(2, 2);
  auto mat1 = Mat(size, Types::CV_32U, &data1[0]);

  std::cout << "\nMat 1:\n";
  mat1.print();

  auto mat2 = Mat(size, Types::CV_32U, &data2[0]);
  std::cout << "\nMat 2:\n";
  mat2.print();

  auto sum = mat1 + mat2;
  std::cout << "\nSum:\n";
  sum.print();

  auto p_sub = mat2 - mat1;
  std::cout << "\nMat 2 - Mat 1:\n";
  p_sub.print();

  // Negative numbers are not handled yet
  auto n_sub = mat1 - mat2;
  std::cout << "\nMat 1 - Mat 2:\n";
  n_sub.print();

  auto dot = mat1 * mat2;
  std::cout << "\nDot Product:\n";
  dot.print();

  auto transpose = mat1.transpose();
  std::cout << "\nTranspose of Mat 1:\n";
  transpose.print();

  auto s_double = mat1 * 5.5;
  std::cout << "\nScalar multiplication of Mat 1 with int value:\n";
  s_double.print();

  return 0;
}

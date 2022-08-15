#ifndef MAT_HPP
#define MAT_HPP

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <string.h>

namespace cv
{
  enum Types {
    CV_NONE = 0,
    CV_8U = 1,
    CV_8UC1 = CV_8U,
    CV_8UC2,
    CV_8UC3,
    CV_8UC4,
    CV_16U,
    CV_16UC1 = CV_16U,
    CV_16UC2,
    CV_16UC3,
    CV_16UC4,
    CV_32U,
    CV_32UC1 = CV_32U,
    CV_32UC2,
    CV_32UC3,
    CV_32UC4,
    CV_32F,
    CV_32FC1 = CV_32F,
    CV_32FC2,
    CV_32FC3,
    CV_32FC4,
    CV_64F,
    CV_64FC1 = CV_64F,
    CV_64FC2,
    CV_64FC3,
    CV_64FC4,
    CV_RGB8 = CV_8UC3,
    CV_BGR8 = CV_8UC3,
    CV_RGBA8 = CV_8UC4,
    CV_BGRA8 = CV_8UC4
  };

  class Size
  {
    public:
      Size();
      Size(uint16_t rows, uint16_t cols);
      size_t count();

      bool operator==(const Size& s);
      bool operator!=(const Size& s);

    public:
      uint16_t rows, cols;
  };

  class Mat
  {
    public:
      Mat() = delete;
      Mat(Size size, Types type, void* data=nullptr);
      Mat(uint16_t rows, uint16_t cols, Types type, void* data=nullptr);
      Mat(Size size, uint8_t data_type, uint8_t channels, void *data = nullptr);
      Mat(uint16_t rows, uint16_t cols, uint8_t data_type, uint8_t channels, void *data = nullptr);

      Mat operator+(const Mat& m);
      Mat operator-(const Mat& m);
      Mat operator*(const Mat& m);
      Mat operator+=(const Mat& m);
      Mat operator-=(const Mat& m);
      Mat operator*=(const Mat& m);

      Mat operator*(const int& scalar);
      Mat operator*(const float& scalar);
      Mat operator*(const double& scalar);

      operator uint16_t();
      operator uint32_t();
      operator float();
      operator double();

      Mat transpose();
      Size size() const;
      Types type() const;
      uint8_t data_type() const;
      uint8_t channels() const;

      void print();

    private:
      Types create_type(uint8_t data_type, uint8_t channels);

      template <typename T>
      void add(T o_data, T c_data, T m_data, Size size);

      template <typename T>
      void sub(T o_data, T c_data, T m_data, Size size);

      template <typename T>
      void mul(T o_data, T c_data, T m_data, Size c_size, Size m_size);

      template <typename T1, typename T2, typename T3>
      void s_mul(T1 o_data, T2 c_data, T3 scalar, Size size);

      template <typename T1, typename T2>
      void cast(T1 *o_data, T2 *i_data);

      template<typename T>
      void prime(T *io_data);

      template <typename T>
      void matrix_display(T o_data);

    public:
      std::vector<uint8_t> data;

    private:
      Size size_;
      Types type_;

      uint8_t data_type_;
      uint8_t channels_;
  };
}

#endif // MAT_HPP

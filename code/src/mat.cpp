#include "mat.hpp"

// TODO: Refactoring required with appropriate design patterns
// First version is for common use only

namespace cv
{

  // MARK: Size class implementation

  Size::Size()
  {
    rows = cols = 0;
  }

  Size::Size(uint16_t rows, uint16_t cols)
  {
    this->rows = rows;
    this->cols = cols;
  }

  size_t Size::count()
  {
    return rows * cols;
  }

  bool Size::operator==(const Size& s)
  {
    return ((rows == s.rows) && (cols == s.cols));
  }

  bool Size::operator!=(const Size& s)
  {
    return ((rows != s.rows) && (cols != s.cols));
  }

  // MARK: Mat class implementation

  Mat::Mat(Size size, Types type, void *data)
  {
    this->size_ = size;
    this->type_ = type;

    size_t length = 0;

    switch(type)
    {
      case CV_8UC1: length = size_.count() * sizeof(uint8_t) * 1; this->data_type_ = 0; this->channels_ = 1; break;
      case CV_8UC2: length = size_.count() * sizeof(uint8_t) * 2; this->data_type_ = 0; this->channels_ = 2; break;
      case CV_8UC3: length = size_.count() * sizeof(uint8_t) * 3; this->data_type_ = 0; this->channels_ = 3; break;
      case CV_8UC4: length = size_.count() * sizeof(uint8_t) * 4; this->data_type_ = 0; this->channels_ = 4; break;
      case CV_16UC1: length = size_.count() * sizeof(uint16_t) * 1; this->data_type_ = 1; this->channels_ = 1; break;
      case CV_16UC2: length = size_.count() * sizeof(uint16_t) * 2; this->data_type_ = 1; this->channels_ = 2; break;
      case CV_16UC3: length = size_.count() * sizeof(uint16_t) * 3; this->data_type_ = 1; this->channels_ = 3; break;
      case CV_16UC4: length = size_.count() * sizeof(uint16_t) * 4; this->data_type_ = 1; this->channels_ = 4; break;
      case CV_32UC1: length = size_.count() * sizeof(uint32_t) * 1; this->data_type_ = 2; this->channels_ = 1; break;
      case CV_32UC2: length = size_.count() * sizeof(uint32_t) * 2; this->data_type_ = 2; this->channels_ = 2; break;
      case CV_32UC3: length = size_.count() * sizeof(uint32_t) * 3; this->data_type_ = 2; this->channels_ = 3; break;
      case CV_32UC4: length = size_.count() * sizeof(uint32_t) * 4; this->data_type_ = 2; this->channels_ = 4; break;
      case CV_8SC1: length = size_.count() * sizeof(int8_t) * 1; this->data_type_ = 3; this->channels_ = 1; break;
      case CV_8SC2: length = size_.count() * sizeof(int8_t) * 2; this->data_type_ = 3; this->channels_ = 2; break;
      case CV_8SC3: length = size_.count() * sizeof(int8_t) * 3; this->data_type_ = 3; this->channels_ = 3; break;
      case CV_8SC4: length = size_.count() * sizeof(int8_t) * 4; this->data_type_ = 3; this->channels_ = 4; break;
      case CV_16SC1: length = size_.count() * sizeof(int16_t) * 1; this->data_type_ = 4; this->channels_ = 1; break;
      case CV_16SC2: length = size_.count() * sizeof(int16_t) * 2; this->data_type_ = 4; this->channels_ = 2; break;
      case CV_16SC3: length = size_.count() * sizeof(int16_t) * 3; this->data_type_ = 4; this->channels_ = 3; break;
      case CV_16SC4: length = size_.count() * sizeof(int16_t) * 4; this->data_type_ = 4; this->channels_ = 4; break;
      case CV_32SC1: length = size_.count() * sizeof(int32_t) * 1; this->data_type_ = 5; this->channels_ = 1; break;
      case CV_32SC2: length = size_.count() * sizeof(int32_t) * 2; this->data_type_ = 5; this->channels_ = 2; break;
      case CV_32SC3: length = size_.count() * sizeof(int32_t) * 3; this->data_type_ = 5; this->channels_ = 3; break;
      case CV_32SC4: length = size_.count() * sizeof(int32_t) * 4; this->data_type_ = 5; this->channels_ = 4; break;
      case CV_32FC1: length = size_.count() * sizeof(float) * 1; this->data_type_ = 6; this->channels_ = 1; break;
      case CV_32FC2: length = size_.count() * sizeof(float) * 2; this->data_type_ = 6; this->channels_ = 2; break;
      case CV_32FC3: length = size_.count() * sizeof(float) * 3; this->data_type_ = 6; this->channels_ = 3; break;
      case CV_32FC4: length = size_.count() * sizeof(float) * 4; this->data_type_ = 6; this->channels_ = 4; break;
      case CV_64FC1: length = size_.count() * sizeof(double) * 1; this->data_type_ = 7; this->channels_ = 1; break;
      case CV_64FC2: length = size_.count() * sizeof(double) * 2; this->data_type_ = 7; this->channels_ = 2; break;
      case CV_64FC3: length = size_.count() * sizeof(double) * 3; this->data_type_ = 7; this->channels_ = 3; break;
      case CV_64FC4: length = size_.count() * sizeof(double) * 4; this->data_type_ = 7; this->channels_ = 4; break;
      default:
        throw std::runtime_error("Type not specified");
    }

    if (data) {
      this->data.reserve(length);

      uint8_t *data_ptr = reinterpret_cast<uint8_t *>(data);
      this->data.insert(this->data.begin(), data_ptr, data_ptr + length);
    }
    else {
      this->data.resize(length);
    }
  }

  Mat::Mat(uint16_t rows, uint16_t cols, Types type, void *data)
  {
    Mat(Size(rows, cols), type, data);
  }


  Mat::Mat(Size size, uint8_t data_type, uint8_t channels, void *data)
  {
    Mat(size, create_type(data_type, channels), data);
  }

  Mat::Mat(uint16_t rows, uint16_t cols, uint8_t data_type, uint8_t channels, void *data)
  {
    Mat(Size(rows, cols), create_type(data_type, channels), data);
  }

  uint8_t Mat::data_type() const
  {
    return data_type_;
  }

  uint8_t Mat::channels() const
  {
    return channels_;
  }

  Types Mat::create_type(uint8_t data_type, uint8_t channels)
  {
    return (Types)((data_type * 4) + channels);
  }

  template <typename T>
  void Mat::add(T o_data, T c_data, T m_data, Size size)
  {
    for (int r = 0; r < size.rows; r++) {
      for (int c = 0; c < size.cols; c++) {
        auto index = (r * size.cols) + c;
        o_data[index] = c_data[index] + m_data[index];
      }
    }
  }

  template <typename T>
  void Mat::sub(T o_data, T c_data, T m_data, Size size)
  {
    for (int r = 0; r < size.rows; r++) {
      for (int c = 0; c < size.cols; c++) {
        auto index = (r * size.cols) + c;
        o_data[index] = c_data[index] - m_data[index];
      }
    }
  }

  template <typename T>
  void Mat::mul(T o_data, T c_data, T m_data, Size c_size, Size m_size)
  {
    auto o_index = 0;
    auto c_index = 0;
    auto m_index = 0;
    for (int c_r = 0, m_c = 0; c_r < c_size.rows; ) {
      for (int c_c = 0, m_r = 0; c_c < c_size.cols; c_c++, m_r++) {
        c_index = (c_r * c_size.cols) + c_c;
        m_index = (m_r * m_size.cols) + m_c;
        o_data[o_index] += (c_data[c_index] * m_data[m_index]);
      }
      o_index++;
      m_c++;
      if (m_c >= m_size.cols) {
        m_c = 0;
        c_r++;
      }
    }
  }

  template <typename T1, typename T2, typename T3>
  void Mat::s_mul(T1 o_data, T2 c_data, T3 scalar, Size size)
  {
    for (int r = 0; r < size.rows; r++) {
      for (int c = 0; c < size.cols; c++) {
        auto index = (r * size.cols) + c;
        o_data[index] = c_data[index] * scalar;
      }
    }
  }

  Mat Mat::operator+(const Mat& m)
  {
    if (m.size() != size_)
    {
      throw std::runtime_error("Addition operation not possible");
    }

    bool is_signed_required = false;
    if ((m.type() != type_) && ((m.type() >= CV_8S) && (m.type() <= CV_32SC4)))
      is_signed_required = true;

    auto mat = Mat(size_, is_signed_required ? create_type(data_type_ + 3, channels_) : type_);

    if (type_ <= CV_8UC4) {
      add((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_16UC4) {
      add((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32UC4) {
      add((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_8SC4) {
      add((int8_t*)&mat.data[0], (int8_t*)&data[0], (int8_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_16SC4) {
      add((int16_t*)&mat.data[0], (int16_t*)&data[0], (int16_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32SC4) {
      add((int32_t*)&mat.data[0], (int32_t*)&data[0], (int32_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32FC4) {
      add((float*)&mat.data[0], (float*)&data[0], (float*)&m.data[0], size_);
    }
    else {
      add((double*)&mat.data[0], (double*)&data[0], (double*)&m.data[0], size_);
    }

    return mat;
  }

  Mat Mat::operator-(const Mat& m)
  {
    if (m.size() != size_)
    {
      throw std::runtime_error("Subtraction operation not possible");
    }

    bool is_signed_required = false;
    if (type_ <= CV_32UC4)
      is_signed_required = true;

    auto mat = Mat(size_, is_signed_required ? create_type(data_type_ + 3, channels_) : type_);

    if (type_ <= CV_8UC4) {
      sub((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_16UC4) {
      sub((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32UC4) {
      sub((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_8SC4) {
      sub((int8_t*)&mat.data[0], (int8_t*)&data[0], (int8_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_16SC4) {
      sub((int16_t*)&mat.data[0], (int16_t*)&data[0], (int16_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32SC4) {
      sub((int32_t*)&mat.data[0], (int32_t*)&data[0], (int32_t*)&m.data[0], size_);
    }
    else if (type_ <= CV_32FC4) {
      sub((float*)&mat.data[0], (float*)&data[0], (float*)&m.data[0], size_);
    }
    else {
      sub((double*)&mat.data[0], (double*)&data[0], (double*)&m.data[0], size_);
    }

    return mat;
  }

  Mat Mat::operator*(const Mat& m)
  {
    if (size_.cols != m.size().rows)
    {
      throw std::runtime_error("Multiplication operation not possible");
    }

    auto new_size = Size(size_.rows, m.size().cols);
    auto mat = Mat(new_size, type_);

    if (type_ <= CV_8UC4) {
      mul((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_16UC4) {
      mul((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_32UC4) {
      mul((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_8SC4) {
      mul((int8_t*)&mat.data[0], (int8_t*)&data[0], (int8_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_16SC4) {
      mul((int16_t*)&mat.data[0], (int16_t*)&data[0], (int16_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_32SC4) {
      mul((int32_t*)&mat.data[0], (int32_t*)&data[0], (int32_t*)&m.data[0], size_, m.size());
    }
    else if (type_ <= CV_32FC4) {
      mul((float*)&mat.data[0], (float*)&data[0], (float*)&m.data[0], size_, m.size());
    }
    else {
      mul((double*)&mat.data[0], (double*)&data[0], (double*)&m.data[0], size_, m.size());
    }

    return mat;
  }

  Mat Mat::operator+=(const Mat& m)
  {
    return *this + m;
  }

  Mat Mat::operator-=(const Mat& m)
  {
    return *this - m;
  }

  Mat Mat::operator*=(const Mat& m)
  {
    return *this * m;
  }

  Mat Mat::operator*(const int& scalar)
  {
    if (type_ <= CV_32UC4) {
      Mat mat(size_, create_type(scalar > 0 ? 2 : 5, channels_));

      if (scalar > 0) {
        if (type_ <= CV_8UC4) {
          s_mul((uint32_t *)&mat.data[0], (uint8_t *)&data[0], scalar, size_);
        }
        else if (type_ <= CV_16UC4) {
          s_mul((uint32_t *)&mat.data[0], (uint16_t *)&data[0], scalar, size_);
        }
        else {
          s_mul((uint32_t *)&mat.data[0], (uint32_t *)&data[0], scalar, size_);
        }
      }
      else {
        if (type_ <= CV_8UC4) {
          s_mul((int32_t *)&mat.data[0], (uint8_t *)&data[0], scalar, size_);
        }
        else if (type_ <= CV_16UC4) {
          s_mul((int32_t *)&mat.data[0], (uint16_t *)&data[0], scalar, size_);
        }
        else {
          s_mul((int32_t *)&mat.data[0], (uint32_t *)&data[0], scalar, size_);
        }
      }
      return mat;
    }
    else if (type_ <= CV_32SC4) {
      Mat mat(size_, create_type(5, channels_));

      if (type_ <= CV_8SC4) {
        s_mul((int32_t *)&mat.data[0], (int8_t *)&data[0], scalar, size_);
      }
      else if (type_ <= CV_16SC4) {
        s_mul((int32_t *)&mat.data[0], (int16_t *)&data[0], scalar, size_);
      }
      else {
        s_mul((int32_t *)&mat.data[0], (int32_t *)&data[0], scalar, size_);
      }
      return mat;
    }
    else {
      Mat mat(size_, type_);
      if (type_ == CV_32FC4) {
        s_mul((float *)&mat.data[0], (float *)&data[0], scalar, size_);
      }
      else {
        s_mul((double *)&mat.data[0], (double *)&data[0], scalar, size_);
      }
      return mat;
    }
  }

  Mat Mat::operator*(const float& scalar)
  {
    if (type_ <= CV_32UC4) {
      Mat mat(size_, create_type(6, channels_));

      if (type_ <= CV_8UC4) {
        s_mul((float *)&mat.data[0], (uint8_t *)&data[0], scalar, size_);
      }
      else if (type_ <= CV_16UC4) {
        s_mul((float *)&mat.data[0], (uint16_t *)&data[0], scalar, size_);
      }
      else if (type_ <= CV_32UC4) {
        s_mul((float *)&mat.data[0], (uint32_t *)&data[0], scalar, size_);
      }
      else {
        s_mul((float *)&mat.data[0], (float *)&data[0], scalar, size_);
      }
      return mat;
    }
    else if (type_ <= CV_32SC4) {
      Mat mat(size_, create_type(6, channels_));

      if (type_ <= CV_8SC4) {
        s_mul((float *)&mat.data[0], (int8_t *)&data[0], scalar, size_);
      }
      else if (type_ <= CV_16SC4) {
        s_mul((float *)&mat.data[0], (int16_t *)&data[0], scalar, size_);
      }
      else if (type_ <= CV_32SC4) {
        s_mul((float *)&mat.data[0], (int32_t *)&data[0], scalar, size_);
      }
      else {
        s_mul((float *)&mat.data[0], (float *)&data[0], scalar, size_);
      }
      return mat;
    }
    else {
      Mat mat(size_, type_);
      s_mul((double *)&mat.data[0], (double *)&data[0], scalar, size_);
      return mat;
    }
  }

  Mat Mat::operator*(const double& scalar)
  {
    Mat mat(size_, create_type(7, channels_));

    if (type_ <= CV_8UC4) {
      s_mul((double *)&mat.data[0], (uint8_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_16UC4) {
      s_mul((double *)&mat.data[0], (uint16_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_32UC4) {
      s_mul((double *)&mat.data[0], (uint32_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_8SC4) {
      s_mul((double *)&mat.data[0], (int8_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_16SC4) {
      s_mul((double *)&mat.data[0], (int16_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_32SC4) {
      s_mul((double *)&mat.data[0], (int32_t *)&data[0], scalar, size_);
    }
    else if (type_ <= CV_32FC4) {
      s_mul((double *)&mat.data[0], (float *)&data[0], scalar, size_);
    }
    else {
      s_mul((double *)&mat.data[0], (double *)&data[0], scalar, size_);
    }

    return mat;
  }

  template <typename T1, typename T2>
  void Mat::cast(T1 *o_data, T2 *i_data)
  {
    for (int r = 0; r < size_.rows; r++) {
      for(int c = 0; c < size_.cols; c++) {
        o_data[(r * size_.cols) + c] = (T1)i_data[(r * size_.cols) + c];
      }
    }
  }

  Mat::operator uint16_t()
  {
    auto m = Mat(size_, create_type(1, channels_));

    uint16_t *o_data = reinterpret_cast<uint16_t*>(&m.data[0]);

    if (type_ <= CV_8UC4) {
      cast(o_data, (uint8_t *)&data[0]);
    }
    else if (type_ <= CV_16UC4) {
      cast(o_data, (uint16_t *)&data[0]);
    }
    else if (type_ <= CV_32UC4) {
      cast(o_data, (uint32_t *)&data[0]);
    }
    else if (type_ <= CV_32FC4) {
      cast(o_data, (float *)&data[0]);
    }
    else {
      cast(o_data, (double *)&data[0]);
    }

    return m;
  }

  Mat::operator uint32_t()
  {
    auto m = Mat(size_, create_type(2, channels_));

    uint32_t *o_data = reinterpret_cast<uint32_t*>(&m.data[0]);

    if (type_ <= CV_8UC4) {
      cast(o_data, (uint8_t *)&data[0]);
    }
    else if (type_ <= CV_16UC4) {
      cast(o_data, (uint16_t *)&data[0]);
    }
    else if (type_ <= CV_32UC4) {
      cast(o_data, (uint32_t *)&data[0]);
    }
    else if (type_ <= CV_32FC4) {
      cast(o_data, (float *)&data[0]);
    }
    else {
      cast(o_data, (double *)&data[0]);
    }

    return m;
  }

  Mat::operator float()
  {
    auto m = Mat(size_, create_type(3, channels_));

    float *o_data = reinterpret_cast<float*>(&m.data[0]);

    if (type_ <= CV_8UC4) {
      cast(o_data, (uint8_t *)&data[0]);
    }
    else if (type_ <= CV_16UC4) {
      cast(o_data, (uint16_t *)&data[0]);
    }
    else if (type_ <= CV_32UC4) {
      cast(o_data, (uint32_t *)&data[0]);
    }
    else if (type_ <= CV_32FC4) {
      cast(o_data, (float *)&data[0]);
    }
    else {
      cast(o_data, (double *)&data[0]);
    }

    return m;
  }

  Mat::operator double()
  {
    auto m = Mat(size_, create_type(4, channels_));

    double *o_data = reinterpret_cast<double*>(&m.data[0]);

    if (type_ <= CV_8UC4) {
      cast(o_data, (uint8_t *)&data[0]);
    }
    else if (type_ <= CV_16UC4) {
      cast(o_data, (uint16_t *)&data[0]);
    }
    else if (type_ <= CV_32UC4) {
      cast(o_data, (uint32_t *)&data[0]);
    }
    else if (type_ <= CV_32FC4) {
      cast(o_data, (float *)&data[0]);
    }
    else {
      cast(o_data, (double *)&data[0]);
    }

    return m;
  }

  template<typename T>
  void Mat::prime(T *io_data)
  {
    std::vector<uint8_t> tmp;
    tmp.resize(data.size());

    T *o_data = reinterpret_cast<T*>(&tmp[0]);

    for (int r = 0; r < size_.rows; r++) {
      for (int c = 0; c < size_.cols; c++) {
        o_data[(c * size_.rows) + r] = io_data[(r * size_.cols) + c];
      }
    }

    memset((void *)io_data, '\0', data.size());
    memcpy((void *)io_data, (void *)o_data, data.size());
  }

  Mat Mat::transpose()
  {
    auto new_size = Size(size_.cols, size_.rows);
    auto m = Mat(new_size, type_, &data[0]);

    if (type_ <= CV_8SC4) {
      prime((int8_t *)&m.data[0]);
    }
    else if (type_ <= CV_16SC4) {
      prime((int16_t *)&m.data[0]);
    }
    else if (type_ <= CV_32SC4) {
      prime((int32_t *)&m.data[0]);
    }
    else if (type_ <= CV_8UC4) {
      prime((uint8_t *)&m.data[0]);
    }
    else if (type_ <= CV_16UC4) {
      prime((uint16_t *)&m.data[0]);
    }
    else if (type_ <= CV_32UC4) {
      prime((uint32_t *)&m.data[0]);
    }
    else if (type_ <= CV_32FC4) {
      prime((float *)&m.data[0]);
    }
    else {
      prime((double *)&m.data[0]);
    }

    return m;
  }

  Size Mat::size() const
  {
    return size_;
  }

  Types Mat::type() const
  {
    return type_;
  }

  template <typename T>
  void Mat::matrix_display(T o_data)
  {
    for (int r = 0; r < size_.rows; r++) {
      for (int c = 0; c < size_.cols; c++) {
        auto index = (r * size_.cols) + c;
        std::cout << o_data[index] << " ";
      }
      std::cout << std::endl;
    }
  }

  void Mat::print()
  {
    if (type_ <= CV_8SC4) {
      matrix_display((int8_t *)&data[0]);
    }
    else if (type_ <= CV_16SC4) {
      matrix_display((int16_t *)&data[0]);
    }
    else if (type_ <= CV_32SC4) {
      matrix_display((int32_t *)&data[0]);
    }
    else if (type_ <= CV_8UC4) {
      matrix_display((uint8_t *)&data[0]);
    }
    else if (type_ <= CV_16UC4) {
      matrix_display((uint16_t *)&data[0]);
    }
    else if (type_ <= CV_32UC4) {
      matrix_display((uint32_t *)&data[0]);
    }
    else if (type_ <= CV_32FC4) {
      matrix_display((float *)&data[0]);
    }
    else {
      matrix_display((double *)&data[0]);
    }
  }
}

#include "mat.hpp"

namespace cv
{
   Mat::Mat(Size size, Types type, void *data)
   {
       size_ = size;
       type_ = type;

       size_t length = 0;

       switch(type)
       {
        case CV_8UC1: length = size_.count() * sizeof(uint8_t) * 1; data_type_ = 0; channels_ = 1; break;                     
        case CV_8UC2: length = size_.count() * sizeof(uint8_t) * 2; data_type_ = 0; channels_ = 2; break;
        case CV_8UC3: length = size_.count() * sizeof(uint8_t) * 3; data_type_ = 0; channels_ = 3; break;
        case CV_8UC4: length = size_.count() * sizeof(uint8_t) * 4; data_type_ = 0; channels_ = 4; break;
        case CV_16UC1: length = size_.count() * sizeof(uint16_t) * 1; data_type_ = 1; channels_ = 1; break;
        case CV_16UC2: length = size_.count() * sizeof(uint16_t) * 2; data_type_ = 1; channels_ = 2; break;
        case CV_16UC3: length = size_.count() * sizeof(uint16_t) * 3; data_type_ = 1; channels_ = 3; break;
        case CV_16UC4: length = size_.count() * sizeof(uint16_t) * 4; data_type_ = 1; channels_ = 4; break;
        case CV_32UC1: length = size_.count() * sizeof(uint32_t) * 1; data_type_ = 2; channels_ = 1; break;
        case CV_32UC2: length = size_.count() * sizeof(uint32_t) * 2; data_type_ = 2; channels_ = 2; break; 
        case CV_32UC3: length = size_.count() * sizeof(uint32_t) * 3; data_type_ = 2; channels_ = 3; break;
        case CV_32UC4: length = size_.count() * sizeof(uint32_t) * 4; data_type_ = 2; channels_ = 4; break;
        case CV_32FC1: length = size_.count() * sizeof(float) * 1; data_type_ = 3; channels_ = 1; break;
        case CV_32FC2: length = size_.count() * sizeof(float) * 2; data_type_ = 3; channels_ = 2; break;
        case CV_32FC3: length = size_.count() * sizeof(float) * 3; data_type_ = 3; channels_ = 3; break;
        case CV_32FC4: length = size_.count() * sizeof(float) * 4; data_type_ = 3; channels_ = 4; break;
        case CV_64FC1: length = size_.count() * sizeof(double) * 1; data_type_ = 4; channels_ = 1; break;
        case CV_64FC2: length = size_.count() * sizeof(double) * 2; data_type_ = 4; channels_ = 2; break;
        case CV_64FC3: length = size_.count() * sizeof(double) * 3; data_type_ = 4; channels_ = 3; break;
        case CV_64FC4: length = size_.count() * sizeof(double) * 4; data_type_ = 4; channels_ = 4; break;
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

   Mat Mat::operator+(const Mat& m)
   {
        if (m.size() != size_ && m.type() != type_)
        {
            throw std::runtime_error("Addition operation not possible");
        }
    
        auto mat = Mat(size_, type_);
        
        if (type_ <= CV_8UC4) {
            add((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_);
        }
        else if (type_ <= CV_16UC4) {
            add((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_);
        }
        else if (type_ <= CV_32UC4) {
            add((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_);
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
        if (m.size() != size_ && m.type() != type_)
        {
            throw std::runtime_error("Subtraction operation not possible");
        }

        auto mat = Mat(size_, type_);
        
        if (type_ <= CV_8UC4) {
            sub((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_);
        }
        else if (type_ <= CV_16UC4) {
            sub((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_);
        }
        else if (type_ <= CV_32UC4) {
            sub((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_);
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

        auto mat = Mat(size_.rows, m.size().cols , type_);
        
        if (type_ <= CV_8UC4) {
            mul((uint8_t*)&mat.data[0], (uint8_t*)&data[0], (uint8_t*)&m.data[0], size_, m.size());
        }
        else if (type_ <= CV_16UC4) {
            mul((uint16_t*)&mat.data[0], (uint16_t*)&data[0], (uint16_t*)&m.data[0], size_, m.size());
        }
        else if (type_ <= CV_32UC4) {
            mul((uint32_t*)&mat.data[0], (uint32_t*)&data[0], (uint32_t*)&m.data[0], size_, m.size());
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
            Mat mat(size_, create_type(2, channels_));
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
        else {
            Mat mat(size_, type_);
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
   }

   Mat Mat::operator*(const float& scalar)
   {
        if (type_ <= CV_32UC4) {
            Mat mat(size_, create_type(3, channels_));
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
        else {
            Mat mat(size_, type_);
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
   }

   Mat Mat::operator*(const double& m)
   {
        if (type_ <= CV_32FC4) {
            Mat mat(size_, create_type(4, channels_));
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
        else {
            Mat mat(size_, type_);
            std::transform(data.begin(), data.end(), mat.data.begin(), [&](auto &val) {
                return val * scalar;
            });
            return mat;
        }
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
      tmp.resize(data.size);

      T *o_data = reinterpret_cast<T*>(&tmp[0]);

      for (int r = 0; r < size_.rows r++) {
        for (int c = 0; c < size_.cols; c++) {    
            o_data[(c * size_.rows) + r] = io_data[(r * size_.cols) + c];
        }
      }

      memset((void *)io_data, '\0', data.size);
      memcpy((void *)io_data, (void *)o_data, data.size);
   }

   Mat Mat::transpose()
   {
        if (type_ <= CV_8UC4) {
            prime((uint8_t *)&data[0]);
        }
        else if (type_ <= CV_16UC4) {
            prime((uint16_t *)&data[0]);
        }
        else if (type_ <= CV_32UC4) {
            prime((uint32_t *)&data[0]);
        }
        else if (type_ <= CV_32FC4) {
            prime((float *)&data[0]);
        }
        else {
            prime((double *)&data[0]);
        }
        
        auto new_size = Size(size_.cols, size_.rows);
        size_ = new_size;
   }

   Size Mat::size() const
   {
        return size_;
   }

   Types Mat::type() const
   {
        return type_;
   }

}
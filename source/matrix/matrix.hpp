#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

template <class T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t count_of_rows = 0;
    size_t count_of_cols = 0;

public:
    Matrix();
    Matrix(size_t count_of_rows, size_t count_of_cols);
    Matrix(const T* const* data, size_t count_of_rows, size_t count_of_cols);
    Matrix(const std::vector<std::vector<T>>& data);
    Matrix<T>& operator &= (const Matrix<T>& other);
    Matrix<T>& operator *= (const T& scalar);
    Matrix<T>& operator += (const Matrix<T>& other);
    Matrix<T>& operator -= (const Matrix<T>& other);

    Matrix transpose() const;
    size_t get_rows_count() const;
    size_t get_cols_count() const;
    const std::vector<std::vector<T>>& get_data() const;
    void reshape(size_t new_count_rows, size_t new_count_cols);

    std::vector<T>& operator [] (size_t index);
    const std::vector<T>& operator [] (size_t index) const;

    template <class U>
    friend std::istream& operator >> (std::istream& is, Matrix<U>& matrix);

    template <class U>
    friend std::ostream& operator << (std::ostream& os, const Matrix<U>& matrix);
};

template <class T>
Matrix<T> operator & (const Matrix<T>& left, const Matrix<T>& right) {
    Matrix<T> result = left;
    result &= right;
    return result;
}

template <class T>
Matrix<T> operator + (const Matrix<T>& left, const Matrix<T>& right) {
    Matrix<T> result = left;
    result += right;
    return result;
}

template <class T>
Matrix<T> operator - (const Matrix<T>& left, const Matrix<T>& right) {
    Matrix<T> result = left;
    result -= right;
    return result;
}

template <class T>
Matrix<T> operator * (const Matrix<T>& matrix, const T& scaler) {
    Matrix<T> result = matrix;
    result *= scaler;
    return result;
}

template <class T>
Matrix<T> operator * (const T& scaler, const Matrix<T>& matrix) {
    Matrix<T> result = matrix;
    result *= scaler;
    return result;
}

template <class T>
Matrix<T>::Matrix() : Matrix(8, 8) {}

template <class T>
Matrix<T>::Matrix(size_t count_of_rows, size_t count_of_cols) {
    this->count_of_rows = count_of_rows;
    this->count_of_cols = count_of_cols;
    this->data = std::vector<std::vector<T>>(
        this->count_of_rows, std::vector<T>(this->count_of_cols, T{})
    );
}

template <class T>
Matrix<T>::Matrix(const T* const* data, size_t count_of_rows, size_t count_of_cols) {
    if (!data) {
        throw std::runtime_error("Nullpt detected");
    }

    this->count_of_rows = count_of_rows;
    this->count_of_cols = count_of_cols;

    for (size_t i = 0; i < this->count_of_rows; i++) {
        if (!data[i]) {
            throw std::runtime_error("Nullpt detected");
        }
    }

    this->data = std::vector<std::vector<T>>(
        this->count_of_rows, std::vector<T>(this->count_of_cols, T{})
    );

    for (size_t i = 0; i < this->count_of_rows; i++) {
        for (size_t j = 0; j < this->count_of_cols; j++) {
            this->data[i][j] = data[i][j];
        }
    }
}

template <class T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& data) {
    this->count_of_rows = data.size();
    this->count_of_cols = data[0].size();
    this->data = data;
}

template <class T>
Matrix<T>& Matrix<T>::operator &= (const Matrix<T>& other) {
    if (this != &other) {
        if (this->count_of_cols != other.get_rows_count()) {
            throw std::runtime_error("Not valid dimensionality");
        }

        size_t new_rows_count = this->count_of_rows;
        size_t new_cols_count = other.get_cols_count();
        Matrix<T> new_matrix;
        new_matrix.reshape(new_rows_count, new_cols_count);

        for (size_t i = 0; i < new_rows_count; i++) {
            for (size_t j = 0; j < new_cols_count; j++) {
                for (size_t k = 0; k < this->count_of_cols; k++) {
                    new_matrix[i][j] += this->data[i][k] * other.get_data()[k][j];
                }
            }
        }
        
        *this = new_matrix;
    }

    return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator *= (const T& scalar) {
    for (size_t i = 0; i < this->count_of_rows; i++) {
        for (size_t j = 0; j < this->count_of_cols; j++) {
            this->data[i][j] *= scalar;
        }
    }
    
    return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& other) {
    if (this != &other) {
        if (this->count_of_rows != other.get_rows_count() ||
            this->count_of_cols != other.get_cols_count()) {
            throw std::runtime_error("Not equal dimensionality");
        }

        for (size_t i = 0; i < this->count_of_rows; i++) {
            for (size_t j = 0; j < this->count_of_cols; j++) {
                this->data[i][j] += other.get_data()[i][j];
            }
        }
    }

    return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& other) {
    if (this != &other) {
        if (this->count_of_rows != other.get_rows_count() ||
            this->count_of_cols != other.get_cols_count()) {
            throw std::runtime_error("Not equal dimensionality");
        }

        for (size_t i = 0; i < this->count_of_rows; i++) {
            for (size_t j = 0; j < this->count_of_cols; j++) {
                this->data[i][j] -= other.get_data()[i][j];
            }
        }
    }

    return *this;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(this->count_of_cols, this->count_of_rows);

    for (size_t i = 0; i < this->count_of_rows; i++) {
        for (size_t j = 0; j < this->count_of_cols; j++) {
            result[j][i] = this->data[i][j];
        }
    }
    
    return result;
}

template <class T>
size_t Matrix<T>::get_rows_count() const {
    return this->count_of_rows;
}

template <class T>
size_t Matrix<T>::get_cols_count() const {
    return this->count_of_cols;
}

template <class T>
const std::vector<std::vector<T>>& Matrix<T>::get_data() const {
    return this->data;
}

template <class T>
void Matrix<T>::reshape(size_t new_count_rows, size_t new_count_cols) {
    *this = Matrix<T>(new_count_rows, new_count_cols);
}

template <class T>
std::vector<T>& Matrix<T>::operator [] (size_t index) {
    if (index >= this->count_of_rows) {
        throw std::out_of_range("Index is out of range");
    }

    return this->data[index];
}

template <class T>
const std::vector<T>& Matrix<T>::operator [] (size_t index) const {
    if (index >= this->count_of_rows) {
        throw std::out_of_range("Index is out of range");
    }

    return this->data[index];
}

template <class U>
std::istream& operator >> (std::istream& is, Matrix<U>& matrix) {
    for (size_t i = 0; i < matrix.count_of_rows; i++) {
        for (size_t j = 0; j < matrix.count_of_cols; j++) {
            is >> matrix.data[i][j];
        }
    }

    return is;
}

template <class U>
std::ostream& operator << (std::ostream& os, const Matrix<U>& matrix) {
    for (size_t i = 0; i < matrix.count_of_rows; i++) {
        for (size_t j = 0; j < matrix.count_of_cols; j++) {
            os << matrix.data[i][j];
            if (j + 1 < matrix.count_of_cols) {
                os << ' ';
            }
        }
        os << '\n';
    }

    return os;
}
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include "core/framework/float16.h"

namespace numpy_io {

// Numpy array header structure
struct NumpyHeader {
    std::string dtype;
    std::vector<size_t> shape;
    bool fortran_order;
};

// Template class for numpy array data
template<typename T>
class NumpyArray {
public:
    std::vector<T> data;
    std::vector<size_t> shape;

    NumpyArray() = default;
    NumpyArray(const std::vector<size_t>& shape) : shape(shape) {
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size);
    }

    size_t size() const {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }

    T& operator[](size_t index) {
        return data[index];
    }

    const T& operator[](size_t index) const {
        return data[index];
    }
};

// Function to get numpy dtype string for C++ types
template<typename T>
std::string get_dtype_string();

// Template specialization declarations
template<> std::string get_dtype_string<float>();
template<> std::string get_dtype_string<double>();
template<> std::string get_dtype_string<int32_t>();
template<> std::string get_dtype_string<int64_t>();
template<> std::string get_dtype_string<uint32_t>();
template<> std::string get_dtype_string<uint64_t>();
template<> std::string get_dtype_string<onnxruntime::MLFloat16>();

// Parse numpy header from string
NumpyHeader parse_header(const std::string& header_str);

// Create numpy header string
std::string create_header(const std::string& dtype, const std::vector<size_t>& shape, bool fortran_order = false);

// Read numpy array from file
template<typename T>
NumpyArray<T> read_numpy_array(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid numpy file format");
    }

    // Read version
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    // Read header length
    uint16_t header_len;
    if (major_version == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len_32;
        file.read(reinterpret_cast<char*>(&header_len_32), 4);
        header_len = static_cast<uint16_t>(header_len_32);
    }

    // Read header
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);

    // Parse header
    NumpyHeader header = parse_header(header_str);

    // Create array
    NumpyArray<T> array(header.shape);

    // Read data
    file.read(reinterpret_cast<char*>(array.data.data()), array.size() * sizeof(T));

    if (!file) {
        throw std::runtime_error("Error reading data from file");
    }

    return array;
}

// Write numpy array to file
template<typename T>
void write_numpy_array(const std::string& filename, const NumpyArray<T>& array) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    // Write magic string
    file.write("\x93NUMPY", 6);

    // Write version (1.0)
    uint8_t major_version = 1, minor_version = 0;
    file.write(reinterpret_cast<const char*>(&major_version), 1);
    file.write(reinterpret_cast<const char*>(&minor_version), 1);

    // Create header
    std::string dtype = get_dtype_string<T>();
    std::string header_str = create_header(dtype, array.shape, false);

    // Pad header to 16-byte boundary
    size_t total_header_size = 10 + header_str.length(); // 6 (magic) + 1 + 1 (version) + 2 (length) + header
    size_t padding = (16 - (total_header_size % 16)) % 16;
    header_str += std::string(padding, ' ');
    header_str += '\n';

    // Write header length
    uint16_t header_len = static_cast<uint16_t>(header_str.length());
    file.write(reinterpret_cast<const char*>(&header_len), 2);

    // Write header
    file.write(header_str.c_str(), header_len);

    // Write data
    file.write(reinterpret_cast<const char*>(array.data.data()), array.size() * sizeof(T));

    if (!file) {
        throw std::runtime_error("Error writing to file");
    }
}

} // namespace numpy_io

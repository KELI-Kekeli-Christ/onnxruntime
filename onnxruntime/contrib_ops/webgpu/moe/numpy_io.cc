#include "numpy_io.h"
#include <sstream>
#include <regex>

namespace numpy_io {

NumpyHeader parse_header(const std::string& header_str) {
    NumpyHeader header;

    // Simple regex-based parser for numpy header
    // Example header: "{'descr': '<f4', 'fortran_order': False, 'shape': (100, 200), }"

    // Extract dtype
    std::regex dtype_regex(R"('descr':\s*'([^']+)')");
    std::smatch dtype_match;
    if (std::regex_search(header_str, dtype_match, dtype_regex)) {
        header.dtype = dtype_match[1].str();
    }

    // Extract fortran_order
    std::regex fortran_regex(R"('fortran_order':\s*(True|False))");
    std::smatch fortran_match;
    if (std::regex_search(header_str, fortran_match, fortran_regex)) {
        header.fortran_order = (fortran_match[1].str() == "True");
    }

    // Extract shape
    std::regex shape_regex(R"('shape':\s*\(([^)]*)\))");
    std::smatch shape_match;
    if (std::regex_search(header_str, shape_match, shape_regex)) {
        std::string shape_str = shape_match[1].str();
        std::stringstream ss(shape_str);
        std::string item;

        while (std::getline(ss, item, ',')) {
            // Remove whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);

            if (!item.empty()) {
                header.shape.push_back(std::stoull(item));
            }
        }
    }

    return header;
}

std::string create_header(const std::string& dtype, const std::vector<size_t>& shape, bool fortran_order) {
    std::stringstream ss;
    ss << "{'descr': '" << dtype << "', 'fortran_order': " << (fortran_order ? "True" : "False") << ", 'shape': (";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }

    if (shape.size() == 1) {
        ss << ",";  // Add comma for single-element tuple
    }

    ss << "), }";
    return ss.str();
}

// Template specializations for get_dtype_string
template<> std::string get_dtype_string<float>() { return "<f4"; }
template<> std::string get_dtype_string<double>() { return "<f8"; }
template<> std::string get_dtype_string<int32_t>() { return "<i4"; }
template<> std::string get_dtype_string<int64_t>() { return "<i8"; }
template<> std::string get_dtype_string<uint32_t>() { return "<u4"; }
template<> std::string get_dtype_string<uint64_t>() { return "<u8"; }
template<> std::string get_dtype_string<onnxruntime::MLFloat16>() { return "<f2"; }

} // namespace numpy_io

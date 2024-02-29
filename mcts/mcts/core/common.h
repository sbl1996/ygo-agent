#ifndef MCTS_CORE_COMMON_H_
#define MCTS_CORE_COMMON_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mcts/core/array.h"

namespace py = pybind11;

template<typename dtype>
py::array_t<dtype> ArrayToNumpy(const Array &a) {
    auto *ptr = new std::shared_ptr<char>(a.SharedPtr());
    auto capsule = py::capsule(ptr, [](void *ptr) {
        delete reinterpret_cast<std::shared_ptr<char> *>(ptr);
    });
    return py::array(a.Shape(), reinterpret_cast<dtype *>(a.Data()), capsule);
}

template<typename dtype>
Array NumpyToArray(const py::array_t<dtype> &arr) {
    using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
    ArrayT arr_t(arr);
    ShapeSpec spec(arr_t.itemsize(),
                   std::vector<int>(arr_t.shape(), arr_t.shape() + arr_t.ndim()));
    return {spec, reinterpret_cast<char *>(arr_t.mutable_data())};
}


template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
    using T = typename Sequence::value_type;
    std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::forward<Sequence>(seq));
    return py::array_t<T>({seq_ptr->size()}, {sizeof(T)}, seq_ptr->data());
}


#endif // MCTS_CORE_COMMON_H_
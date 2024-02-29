#ifndef MCTS_CORE_ARRAY_H_
#define MCTS_CORE_ARRAY_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "mcts/core/spec.h"

class Array {
public:
    std::size_t size;
    std::size_t ndim;
    std::size_t element_size;

protected:
    std::vector<std::size_t> shape_;
    std::shared_ptr<char> ptr_;

    template<class Shape, class Deleter>
    Array(char *ptr, Shape &&shape, std::size_t element_size,// NOLINT
          Deleter &&deleter)
        : size(Prod(shape.data(), shape.size())),
          ndim(shape.size()),
          element_size(element_size),
          shape_(std::forward<Shape>(shape)),
          ptr_(ptr, std::forward<Deleter>(deleter)) {}

    template<class Shape>
    Array(std::shared_ptr<char> ptr, Shape &&shape, std::size_t element_size)
        : size(Prod(shape.data(), shape.size())),
          ndim(shape.size()),
          element_size(element_size),
          shape_(std::forward<Shape>(shape)),
          ptr_(std::move(ptr)) {}

public:
    Array() = default;

    /**
   * Constructor an `Array` of shape defined by `spec`, with `data` as pointer
   * to its raw memory. With an empty deleter, which means Array does not own
   * the memory.
   */
    template<class Deleter>
    Array(const ShapeSpec &spec, char *data, Deleter &&deleter)// NOLINT
        : Array(data, spec.Shape(), spec.element_size,
                std::forward<Deleter>(deleter)) {}

    Array(const ShapeSpec &spec, char *data)
        : Array(data, spec.Shape(), spec.element_size, [](char * /*unused*/) {}) {}

    /**
   * Constructor an `Array` of shape defined by `spec`. This constructor
   * allocates and owns the memory.
   */
    explicit Array(const ShapeSpec &spec)
        : Array(spec, nullptr, [](char * /*unused*/) {}) {
        ptr_.reset(new char[size * element_size](),
                   [](const char *p) { delete[] p; });
    }

    /**
   * Take multidimensional index into the Array.
   */
    template<typename... Index>
    inline Array operator()(Index... index) const {
        constexpr std::size_t num_index = sizeof...(Index);
        std::size_t offset = 0;
        std::size_t i = 0;
        for (((offset = offset * shape_[i++] + index), ...); i < ndim; ++i) {
            offset *= shape_[i];
        }
        return Array(
            ptr_.get() + offset * element_size,
            std::vector<std::size_t>(shape_.begin() + num_index, shape_.end()),
            element_size, [](char * /*unused*/) {});
    }

    /**
   * Index operator of array, takes the index along the first axis.
   */
    inline Array operator[](int index) const { return this->operator()(index); }

    /**
   * Take a slice at the first axis of the Array.
   */
    [[nodiscard]] Array Slice(std::size_t start, std::size_t end) const {
        std::vector<std::size_t> new_shape(shape_);
        new_shape[0] = end - start;
        std::size_t offset = 0;
        if (shape_[0] > 0) {
            offset = start * size / shape_[0];
        }
        return {ptr_.get() + offset * element_size, std::move(new_shape),
                element_size, [](char *p) {}};
    }

    /**
   * Copy the content of another Array to this Array.
   */
    void Assign(const Array &value) const {
        std::memcpy(ptr_.get(), value.ptr_.get(), size * element_size);
    }

    /**
   * Return a clone of this array.
   */
    Array Clone() const {
        std::vector<int> shape;
        for (int i = 0; i < ndim; i++) {
            shape.push_back(shape_[i]);
        }
        auto spec = ShapeSpec(element_size, shape);
        Array ret(spec);
        ret.Assign(*this);
        return ret;
    }

    /**
   * Assign to this Array a scalar value. This Array needs to have a scalar
   * shape.
   */
    template<typename T>
    void operator=(const T &value) const {
        *reinterpret_cast<T *>(ptr_.get()) = value;
    }

    /**
   * Fills this array with a scalar value of type T.
   */
    template<typename T>
    void Fill(const T &value) const {
        auto *data = reinterpret_cast<T *>(ptr_.get());
        std::fill(data, data + size, value);
    }

    /**
   * Copy the memory starting at `raw.first`, to `raw.first + raw.second` to the
   * memory of this Array.
   */
    template<typename T>
    void Assign(const T *buff, std::size_t sz) const {
        std::memcpy(ptr_.get(), buff, sz * sizeof(T));
    }

    template<typename T>
    void Assign(const T *buff, std::size_t sz, ptrdiff_t offset) const {
        offset = offset * (element_size / sizeof(char));
        std::memcpy(ptr_.get() + offset, buff, sz * sizeof(T));
    }

    /**
   * Cast the Array to a scalar value of type `T`. This Array needs to have a
   * scalar shape.
   */
    template<typename T>
    operator const T &() const {// NOLINT
        return *reinterpret_cast<T *>(ptr_.get());
    }

    /**
   * Cast the Array to a scalar value of type `T`. This Array needs to have a
   * scalar shape.
   */
    template<typename T>
    operator T &() {// NOLINT
        return *reinterpret_cast<T *>(ptr_.get());
    }

    /**
   * Size of axis `dim`.
   */
    [[nodiscard]] inline std::size_t Shape(std::size_t dim) const {
        return shape_[dim];
    }

    /**
   * Shape
   */
    [[nodiscard]] inline const std::vector<std::size_t> &Shape() const {
        return shape_;
    }

    /**
   * Pointer to the raw memory.
   */
    [[nodiscard]] inline void *Data() const { return ptr_.get(); }

    /**
   * Truncate the Array. Return a new Array that shares the same memory
   * location but with a truncated shape.
   */
    [[nodiscard]] Array Truncate(std::size_t end) const {
        auto new_shape = std::vector<std::size_t>(shape_);
        new_shape[0] = end;
        Array ret(ptr_, std::move(new_shape), element_size);
        return ret;
    }

    void Zero() const { std::memset(ptr_.get(), 0, size * element_size); }
    [[nodiscard]] std::shared_ptr<char> SharedPtr() const { return ptr_; }
};

template<typename Dtype>
class TArray : public Array {
public:
    explicit TArray(const Spec<Dtype> &spec) : Array(spec) {}
};

#endif // MCTS_CORE_ARRAY_H_
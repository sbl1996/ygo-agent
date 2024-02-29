#ifndef MCTS_CORE_SPEC_H_
#define MCTS_CORE_SPEC_H_

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

static std::size_t Prod(const std::size_t *shape, std::size_t ndim) {
    return std::accumulate(shape, shape + ndim, static_cast<std::size_t>(1),
                           std::multiplies<>());
}

class ShapeSpec {
public:
    int element_size;
    std::vector<int> shape;
    ShapeSpec() = default;
    ShapeSpec(int element_size, std::vector<int> shape_vec)
        : element_size(element_size), shape(std::move(shape_vec)) {}
    [[nodiscard]] ShapeSpec Batch(int batch_size) const {
        std::vector<int> new_shape = {batch_size};
        new_shape.insert(new_shape.end(), shape.begin(), shape.end());
        return {element_size, std::move(new_shape)};
    }
    [[nodiscard]] std::vector<std::size_t> Shape() const {
        auto s = std::vector<std::size_t>(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) {
            s[i] = shape[i];
        }
        return s;
    }
};


template<typename D>
class Spec : public ShapeSpec {
public:
    using dtype = D;// NOLINT
    std::tuple<dtype, dtype> bounds = {std::numeric_limits<dtype>::min(),
                                       std::numeric_limits<dtype>::max()};
    std::tuple<std::vector<dtype>, std::vector<dtype>> elementwise_bounds;
    explicit Spec(std::vector<int> &&shape)
        : ShapeSpec(sizeof(dtype), std::move(shape)) {}
    explicit Spec(const std::vector<int> &shape)
        : ShapeSpec(sizeof(dtype), shape) {}

    /* init with constant bounds */
    Spec(std::vector<int> &&shape, std::tuple<dtype, dtype> &&bounds)
        : ShapeSpec(sizeof(dtype), std::move(shape)), bounds(std::move(bounds)) {}
    Spec(const std::vector<int> &shape, const std::tuple<dtype, dtype> &bounds)
        : ShapeSpec(sizeof(dtype), shape), bounds(bounds) {}

    /* init with elementwise bounds */
    Spec(std::vector<int> &&shape,
         std::tuple<std::vector<dtype>, std::vector<dtype>> &&elementwise_bounds)
        : ShapeSpec(sizeof(dtype), std::move(shape)),
          elementwise_bounds(std::move(elementwise_bounds)) {}
    Spec(const std::vector<int> &shape,
         const std::tuple<std::vector<dtype>, std::vector<dtype>> &
             elementwise_bounds)
        : ShapeSpec(sizeof(dtype), shape),
          elementwise_bounds(elementwise_bounds) {}

    [[nodiscard]] Spec Batch(int batch_size) const {
        std::vector<int> new_shape = {batch_size};
        new_shape.insert(new_shape.end(), shape.begin(), shape.end());
        return Spec(std::move(new_shape));
    }
};

template<typename dtype>
class TArray;

template<typename dtype>
using Container = std::unique_ptr<TArray<dtype>>;

template<typename D>
class Spec<Container<D>> : public ShapeSpec {
public:
    using dtype = Container<D>;// NOLINT
    Spec<D> inner_spec;
    explicit Spec(const std::vector<int> &shape, const Spec<D> &inner_spec)
        : ShapeSpec(sizeof(Container<D>), shape), inner_spec(inner_spec) {}
    explicit Spec(std::vector<int> &&shape, Spec<D> &&inner_spec)
        : ShapeSpec(sizeof(Container<D>), std::move(shape)),
          inner_spec(std::move(inner_spec)) {}
};

#endif // MCTS_CORE_SPEC_H_
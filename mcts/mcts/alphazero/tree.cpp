#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>

#include "mcts/core/common.h"
#include "mcts/core/minimax.h"
#include "mcts/core/array.h"

#include "mcts/alphazero/cnode.h"

namespace py = pybind11;

PYBIND11_MODULE(alphazero_mcts, m) {
    using namespace pybind11::literals;

    py::class_<MinMaxStatsList>(m, "MinMaxStatsList")
    .def(py::init<int>())
    .def("set_delta", &MinMaxStatsList::set_delta);

    py::class_<tree::SearchResults>(m, "SearchResults")
    .def(py::init<int>())
    .def("get_search_len", [](tree::SearchResults &results) {
        return results.search_lens;
    });

    py::class_<tree::Roots>(m, "Roots")
    .def(py::init<int>())
    .def_readonly("num", &tree::Roots::root_num)
    .def("prepare", [](
        tree::Roots &roots, const py::array_t<float> &rewards,
        const py::array_t<float> &logits, const py::array_t<int> &all_legal_actions,
        const py::array_t<int> &n_legal_actions, float exploration_fraction,
        float dirichlet_alpha) {
        Array rewards_ = NumpyToArray(rewards);
        Array logits_ = NumpyToArray(logits);
        Array all_legal_actions_ = NumpyToArray(all_legal_actions);
        Array n_legal_actions_ = NumpyToArray(n_legal_actions);
        roots.prepare(rewards_, logits_, all_legal_actions_, n_legal_actions_, exploration_fraction, dirichlet_alpha);
    })
    .def("get_distributions", &tree::Roots::get_distributions)
    .def("get_values", &tree::Roots::get_values);

    m.def("batch_expand", [](
        int state_index, const py::array_t<bool> &game_over, const py::array_t<float> &rewards, const py::array_t<float> &logits,
        const py::array_t<int> &all_legal_actions, const py::array_t<int> &n_legal_actions, tree::SearchResults &results) {
        Array game_over_ = NumpyToArray(game_over);
        Array rewards_ = NumpyToArray(rewards);
        Array logits_ = NumpyToArray(logits);
        Array all_legal_actions_ = NumpyToArray(all_legal_actions);
        Array n_legal_actions_ = NumpyToArray(n_legal_actions);
        tree::batch_expand(state_index, game_over_, rewards_, logits_, all_legal_actions_, n_legal_actions_, results);
    });

    m.def("batch_backpropagate", [](
        float discount_factor, const py::array_t<float> &values,
        MinMaxStatsList &min_max_stats_lst, tree::SearchResults &results) {
        Array values_ = NumpyToArray(values);
        tree::batch_backpropagate(discount_factor, values_, min_max_stats_lst, results);
    });

    m.def("batch_traverse", [](
        tree::Roots &roots, int pb_c_base, float pb_c_init, float discount_factor,
        MinMaxStatsList &min_max_stats_lst, tree::SearchResults &results) {
        tree::batch_traverse(roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results);
        return py::make_tuple(results.state_index_in_search_path, results.state_index_in_batch, results.last_actions);
    });

    m.def("init_module", &tree::init_module, "", "seed"_a);
}
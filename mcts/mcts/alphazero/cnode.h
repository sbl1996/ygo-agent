#ifndef AZ_CNODE_H
#define AZ_CNODE_H

#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <random>
#include <sys/timeb.h>
#include <ctime>
#include <map>

#include "mcts/core/minimax.h"
#include "mcts/core/array.h"

const int DEBUG_MODE = 0;

namespace tree {

    void init_module(int seed);

    using Action = int;

    class Node {
        public:
            int visit_count, state_index, batch_index, best_action;
            float reward, prior, value_sum;
            std::map<Action, Node> children;

            Node();
            Node(float prior);
            ~Node();

            void expand(
                int state_index, int batch_index, float reward, const Array &logits, const Array &legal_actions);
            void add_exploration_noise(float exploration_fraction, float dirichlet_alpha);
            float compute_mean_q(int isRoot, float parent_q, float discount_factor);

            int expanded() const;
            float value() const;
            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            Node* get_child(int action);
    };

    class Roots{
        public:
            int root_num;
            std::vector<Node> roots;

            Roots();
            Roots(int root_num);
            ~Roots();

            void prepare(
                const Array &rewards, const Array &logits,
                const Array &all_legal_actions, const Array &n_legal_actions,
                float exploration_fraction, float dirichlet_alpha);
            void clear();
            std::vector<std::vector<int> > get_trajectories();
            std::vector<std::vector<int> > get_distributions();
            std::vector<float> get_values();

    };

    class SearchResults{
        public:
            int num;
            std::vector<int> state_index_in_search_path, state_index_in_batch, last_actions, search_lens;
            std::vector<Node*> nodes;
            std::vector<std::vector<Node*> > search_paths;

            SearchResults();
            SearchResults(int num);
            ~SearchResults();

    };


    void update_tree_q(Node* root, MinMaxStats &min_max_stats, float discount_factor);
    void backpropagate(std::vector<Node*> &search_path, MinMaxStats &min_max_stats, float value, float discount_factor);
    void batch_expand(
        int state_index, const Array &game_over, const Array &rewards, const Array &logits /* 2D array */,
        const Array &all_legal_actions, const Array &n_legal_actions, SearchResults &results);
    void batch_backpropagate(float discount_factor, const Array &values, MinMaxStatsList &min_max_stats_lst, SearchResults &results);
    int select_child(Node* root, const MinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q);
    float ucb_score(const Node &child, const MinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor);
    void batch_traverse(Roots &roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList &min_max_stats_lst, SearchResults &results);
}

#endif  // AZ_CNODE_H
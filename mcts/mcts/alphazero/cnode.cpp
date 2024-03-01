#include <algorithm>
#include <map>
#include <cassert>

#include "mcts/alphazero/cnode.h"


namespace tree
{

    std::mt19937 rng_ = std::mt19937(time(NULL));
    void init_module(int seed) {
        rng_ = std::mt19937(seed);
    }

    template <class RealType>
    std::vector<RealType> random_dirichlet(RealType alpha, int n) {
        std::gamma_distribution<RealType> gamma(alpha, 1);
        std::vector<RealType> x(n);
        RealType sum = 0.0;
        for (int i = 0; i < n; i++){
            x[i] = gamma(rng_);
            sum += x[i];
        }
        for (int i = 0; i < n; i++) {
            x[i] = x[i] / sum;
        }
        return x;
    }    

    SearchResults::SearchResults()
    {
        /*
        Overview:
            Initialization of SearchResults, the default result number is set to 0.
        */
        this->num = 0;
    }

    SearchResults::SearchResults(int num)
    {
        /*
        Overview:
            Initialization of SearchResults with result number.
        */
        this->num = num;
        for (int i = 0; i < num; ++i)
        {
            this->search_paths.push_back(std::vector<Node *>());
        }
    }

    SearchResults::~SearchResults() {}

    //*********************************************************

    Node::Node()
    {
        /*
        Overview:
            Initialization of Node.
        */
        this->prior = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->reward = 0.0;
    }

    Node::Node(float prior)
    {
        /*
        Overview:
            Initialization of Node with prior value and legal actions.
        Arguments:
            - prior: the prior value of this node.
            - legal_actions: a vector of legal actions of this node.
        */
        this->prior = prior;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->reward = 0.0;

        this->state_index = -1;
        this->batch_index = -1;
    }

    Node::~Node() {}

    void Node::expand(
        int state_index, int batch_index, float reward, const Array &logits, const Array &legal_actions)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - state_index: The index of state of the leaf node in the search path of the current node.
            - batch_index: The index of state of the leaf node in the search path of the current node.
            - reward: the reward of the current node.
            - logits: the logit of the child nodes.
        */
        this->state_index = state_index;
        this->batch_index = batch_index;
        this->reward = reward;

        float temp_policy;
        float policy_sum = 0.0;

        int n_actions = logits.Shape(0);
        int n_legal_actions = legal_actions.Shape(0);
        // Softmax over logits of legal actions
        float policy[n_actions];

        float policy_max = FLOAT_MIN;
        for (int i = 0; i < n_legal_actions; ++i)
        {
            int a = legal_actions[i];
            float logit = logits[a];
            if (policy_max < logit)
            {
                policy_max = logit;
            }
        }

        for (int i = 0; i < n_legal_actions; ++i)
        {
            int a = legal_actions[i];
            float logit = logits[a];
            temp_policy = exp(logit - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        for (int i = 0; i < n_legal_actions; ++i)
        {
            int a = legal_actions[i];
            prior = policy[a] / policy_sum;
            this->children[a] = Node(prior);
        }
    }

    void Node::add_exploration_noise(float exploration_fraction, float dirichlet_alpha)
    {
        /*
        Overview:
            Add a noise to the prior of the child nodes.
        Arguments:
            - exploration_fraction: the fraction to add noise.
            - noises: the vector of noises added to each child node.
        */
        std::vector<float> noises = random_dirichlet(dirichlet_alpha, this->children.size());
        float noise, prior;
        int i = 0;
        for (auto &[a, child] : this->children)
        {
            noise = noises[i++];
            prior = child.prior;
            child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float Node::compute_mean_q(int isRoot, float parent_q, float discount_factor)
    {
        /*
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - isRoot: whether the current node is a root node.
            - parent_q: the q value of the parent node.
            - discount_factor: the discount_factor of reward.
        */
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        for (const auto &[a, child] : this->children)
        {
            if (child.visit_count > 0)
            {
                float qsa = child.reward + discount_factor * child.value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if (isRoot && total_visits > 0)
        {
            mean_q = total_unsigned_q / total_visits;
        }
        else
        {
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    int Node::expanded() const
    {
        /*
        Overview:
            Return whether the current node is expanded.
        */
        return this->children.size() > 0;
    }

    float Node::value() const
    {
        /*
        Overview:
            Return the real value of the current tree.
        */
        float true_value = 0.0;
        if (this->visit_count == 0)
        {
            return true_value;
        }
        else
        {
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<int> Node::get_trajectory()
    {
        /*
        Overview:
            Find the current best trajectory starts from the current node.
        Outputs:
            - traj: a vector of node index, which is the current best trajectory from this node.
        */
        std::vector<int> traj;

        Node *node = this;
        int best_action = node->best_action;
        while (best_action >= 0)
        {
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> Node::get_children_distribution()
    {
        /*
        Overview:
            Get the distribution of child nodes in the format of visit_count.
        Outputs:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<int> distribution;
        if (this->expanded())
        {
            for (const auto &[a, child] : this->children)
            {
                distribution.push_back(child.visit_count);
            }
        }
        return distribution;
    }

    Node *Node::get_child(int action)
    {
        /*
        Overview:
            Get the child node corresponding to the input action.
        Arguments:
            - action: the action to get child.
        */
        auto it = this->children.find(action);
        if (it != this->children.end())
        {
            // The action exists in the map, return a pointer to the corresponding Node.
            return &(it->second);
        }
        else
        {
            throw std::out_of_range("Action not found in children");
        }
    }

    //*********************************************************

    Roots::Roots()
    {
        /*
        Overview:
            The initialization of Roots.
        */
        this->root_num = 0;
    }

    Roots::Roots(int root_num)
    {
        /*
        Overview:
            The initialization of Roots with root num and legal action lists.
        Arguments:
            - root_num: the number of the current root.
        */
        this->root_num = root_num;

        for (int i = 0; i < root_num; ++i)
        {
            // root node has no prior
            this->roots.push_back(Node(0));
        }
    }

    Roots::~Roots() {}

    void Roots::prepare(
        const Array &rewards, const Array &logits,
        const Array &all_legal_actions, const Array &n_legal_actions,
        float exploration_fraction, float dirichlet_alpha)
    {
        /*
        Overview:
            Expand the roots and add noises.
        Arguments:
            - rewards: the vector of rewards of each root.
            - logits: the vector of policy logits of each root.
            - legal_actions_list: the vector of legal actions of each root.
            - exploration_fraction: the fraction to add noise, 0 means no noise.
            - dirichlet_alpha: the dirichlet alpha.
        Note:
            Do not include terminal states because they have no legal actions and cannot be expanded. 
        */
        int batch_size = this->root_num;
        int offset = 0;
        int n_actions = logits.Shape(1);
        for (int i = 0; i < batch_size; ++i)
        {
            int n_legal_action = n_legal_actions[i];
            const Array &legal_actions = all_legal_actions.Slice(offset, offset + n_legal_action);
            this->roots[i].expand(
                0, i, rewards[i], logits[i], legal_actions);
            if (exploration_fraction > 0) {
                this->roots[i].add_exploration_noise(exploration_fraction, dirichlet_alpha);
            }
            this->roots[i].visit_count += 1;
            offset += n_legal_action;
        }
    }

    void Roots::clear()
    {
        /*
        Overview:
            Clear the roots vector.
        */
        this->roots.clear();
    }

    std::vector<std::vector<int> > Roots::get_trajectories()
    {
        /*
        Overview:
            Find the current best trajectory starts from each root.
        Outputs:
            - traj: a vector of node index, which is the current best trajectory from each root.
        */
        std::vector<std::vector<int> > trajs;
        trajs.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > Roots::get_distributions()
    {
        /*
        Overview:
            Get the children distribution of each root.
        Outputs:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> Roots::get_values()
    {
        /*
        Overview:
            Return the real value of each root.
        */
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(roots[i].value());
        }
        return values;
    }

    //*********************************************************
    //
    void update_tree_q(Node *root, MinMaxStats &min_max_stats, float discount_factor)
    {
        /*
        Overview:
            Update the q value of the root and its child nodes.
        Arguments:
            - root: the root that update q value from.
            - min_max_stats: a tool used to min-max normalize the q value.
            - discount_factor: the discount factor of reward.
        */
        std::stack<Node *> node_stack;
        node_stack.push(root);
        while (node_stack.size() > 0)
        {
            Node *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                float true_reward = node->reward;
                float qsa;
                qsa = true_reward + discount_factor * node->value();

                min_max_stats.update(qsa);
            }

            for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                Node *child = &(it->second);
                if (child->expanded()) {
                    node_stack.push(child);
                }
            }
        }
    }

    void backpropagate(std::vector<Node *> &search_path, MinMaxStats &min_max_stats, float value, float discount_factor)
    {
        /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
        */
        float bootstrap_value = value;
        int path_len = search_path.size();
        for (int i = path_len - 1; i >= 0; --i)
        {
            Node *node = search_path[i];
            node->value_sum += bootstrap_value;
            node->visit_count += 1;

            float true_reward = node->reward;

            min_max_stats.update(true_reward + discount_factor * node->value());

            bootstrap_value = true_reward + discount_factor * bootstrap_value;
        }
    }

    void batch_expand(
        int state_index, const Array &game_over, const Array &rewards, const Array &logits /* 2D array */,
        const Array &all_legal_actions, const Array &n_legal_actions, SearchResults &results)
    {
        int batch_size = results.num;
        int offset = 0;
        int n_actions = logits.Shape(1);
        for (int i = 0; i < batch_size; ++i)
        {
            Node *node = results.nodes[i];
            int n_legal_action = n_legal_actions[i];
            if (game_over[i]) {
                node->state_index = state_index;
                node->batch_index = i;
                node->reward = rewards[i];
            }
            else {
                const Array &legal_actions = all_legal_actions.Slice(offset, offset + n_legal_action);
                node->expand(
                    state_index, i, rewards[i], logits[i], legal_actions);
            }
            offset += n_legal_action;
        }
    }

    void batch_backpropagate(float discount_factor, const Array &values, MinMaxStatsList &min_max_stats_lst, SearchResults &results)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - state_index: The index of state of the leaf node in the search path.
            - values: the values to propagate along the search path.
            - logits: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
        */
        for (int i = 0; i < results.num; ++i)
        {
            backpropagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], values[i], discount_factor);
        }
    }

    int select_child(Node *root, const MinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q)
    {
        /*
        Overview:
            Select the child node of the roots according to ucb scores.
        Arguments:
            - root: the roots to select the child node.
            - min_max_stats: a tool used to min-max normalize the score.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - mean_q: the mean q value of the parent node.
        Outputs:
            - action: the action to select.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<Action> max_index_lst;
        int action = 0;
        for (const auto &[a, child] : root->children)
        {
            action = a;

            float temp_score = ucb_score(child, min_max_stats, mean_q, root->visit_count, pb_c_base, pb_c_init, discount_factor);
            if (max_score < temp_score)
            {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(a);
            }
        }

        if (max_index_lst.size() > 0)
        {
            std::uniform_int_distribution<int> dist(0, max_index_lst.size() - 1);
            int rand_index = dist(rng_);
            action = max_index_lst[rand_index];
        }
        return action;
    }

    float ucb_score(const Node &child, const MinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - mean_q: the mean q value of the parent node.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
        Outputs:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child.visit_count + 1));

        prior_score = pb_c * child.prior;
        if (child.visit_count == 0) {
            value_score = parent_mean_q;
        }
        else {
            value_score = child.reward + discount_factor * child.value();
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;
        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    void batch_traverse(Roots &roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList &min_max_stats_lst, SearchResults &results)
    {
        /*
        Overview:
            Search node path from the roots.
        Arguments:
            - roots: the roots that search from.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - min_max_stats: a tool used to min-max normalize the score.
            - results: the search results.
        */
        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        for (int i = 0; i < results.num; ++i)
        {
            Node *node = &(roots.roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                is_root = 0;
                parent_q = mean_q;

                int action = select_child(node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q);

                node->best_action = action;
                // next
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            Node *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.state_index_in_search_path.push_back(parent->state_index);
            results.state_index_in_batch.push_back(parent->batch_index);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }

}
from typing import List, Tuple, Union

import numpy as np

from . import alphazero_mcts as tree

# ==============================================================
# AlphaZero
# ==============================================================


def select_action(
        visit_counts: np.ndarray,
        temperature: float = 1.0,
        deterministic: bool = True
    ):
    """
    Select action from visit counts of the root node.

    Parameters
    ----------
    visit_counts: np.ndarray, shape (n_legal_actions,)
        The visit counts of the root node.
    temperature: float, default: 1.0
        The temperature used to adjust the sampling distribution.
    deterministic: bool, default: True
        Whether to enable deterministic mode in action selection. True means to
        select the argmax result, False indicates to sample action from the distribution.

    Returns
    -------
    action_pos: np.int64
        The selected action position (index).
    """
    if deterministic:
        action_pos = np.argmax(visit_counts)
    else:
        if temperature != 1:
            visit_counts = visit_counts ** (1 / temperature)
        action_probs = visit_counts / np.sum(visit_counts)
        action_pos = np.random.choice(len(visit_counts), p=action_probs)
    return action_pos


class AlphaZeroMCTSCtree(object):
    """
    MCTSCtree for AlphaZero. The core ``batch_traverse``, ``batch_expand`` and ``batch_backpropagate`` function is implemented in C++.

    Interfaces
    ----------
    __init__, tree_search

    """

    def __init__(
        self,
        env,
        predict_fn=None,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652,
        discount_factor=0.99,
        value_delta_max=0.01,
        seed: int = 0,
        ):
        """
        Initialize the MCTSCtree for AlphaZero.

        Parameters
        ----------
        env:
            The game model.
        predict_fn: Callable[Obs, [np.ndarray, np.ndarray]]
            The function used to predict the policy and value.
        root_dirichlet_alpha: float, optional, default: 0.25
            The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_exploration_fraction: float, default: 0.25
            The noise weight at the root node of the search tree.
        pb_c_init: float, default: 1.25
            The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base: int, default: 19652
            The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        discount_factor: float, default: 0.99
            The discount factor used to calculate the return.
        value_delta_max: float, default: 0.01
            The maximum change in value allowed during the backup step of the search tree update.
        seed: int, default: 0
            The random seed.        
        """
        self._env = env
        self._predict_fn = predict_fn
        self._root_dirichlet_alpha = root_dirichlet_alpha
        self._root_exploration_fraction = root_exploration_fraction
        self._pb_c_init = pb_c_init
        self._pb_c_base = pb_c_base
        self._discount_factor = discount_factor
        self._value_delta_max = value_delta_max

        tree.init_module(seed)

    def set_predict_fn(self, predict_fn):
        self._predict_fn = predict_fn

    def tree_search(
        self,
        init_states,
        num_simulations: int,
        temperature: float = 1.0,
        root_exploration_fraction: float = None,
        sample: bool = False,
        return_roots: bool = False
        ) -> Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, tree.Roots]]:
        """
        Perform MCTS for a batch of root nodes in parallel using the cpp ctree.

        Parameters
        ----------
        init_states : State, shape (parallel_episodes,)
            The states of the roots.
        num_simulations : int
            The number of simulations to run for each root.
        temperature : float, default: 1.0
            The temperature used to adjust the sampling distribution.
        sample : bool, default: False
            Whether to sample action for acting. If False, select the argmax result.
        return_roots : bool, default: False
            Whether to return the roots.

        Returns
        -------
        probs : Tuple[np.ndarray], shape (parallel_episodes, action_dim1), (parallel_episodes, action_dim2), ...
            The target action probabilities of the roots for learning.
        values : np.ndarray, shape (parallel_episodes,)
            The target Q values of the roots.
        action : np.ndarray, shape (parallel_episodes, n_actions)
            The selected action of the roots for acting.
        roots : Roots, optional
            The roots after search. Only returned if return_roots is True.
        """
        assert self._predict_fn is not None, "The predict function is not set."
        if root_exploration_fraction is None:
            root_exploration_fraction = self._root_exploration_fraction

        batch_size = len(init_states)  # parallel_episodes
        obs, all_legal_actions, n_legal_actions = self._env.observation(init_states)
        legal_actions_list = []
        offset = 0
        for i in range(batch_size):
            legal_actions_list.append(all_legal_actions[offset: offset + n_legal_actions[i]].tolist())
            offset += n_legal_actions[i]

        logits, pred_values = self._predict_fn(obs)

        game_over, rewards = self._env.terminal(init_states)

        init_legal_actions_list = legal_actions_list

        roots = tree.Roots(batch_size)
        roots.prepare(rewards, logits, all_legal_actions, n_legal_actions, root_exploration_fraction, self._root_dirichlet_alpha)

        # the data storage of states: storing the state of all the nodes in the search.
        # shape: (num_simulations, batch_size)
        state_batch_in_search_path = [init_states]

        # minimax value storage
        min_max_stats_lst = tree.MinMaxStatsList(batch_size)
        min_max_stats_lst.set_delta(self._value_delta_max)

        for i_sim in range(1, num_simulations + 1):
            # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

            # prepare a result wrapper to transport results between python and c++ parts
            results = tree.SearchResults(batch_size)

            # state_index_in_search_path: the first index of leaf node states in state_batch_in_search_path, i.e. is state_index in one the search.
            # state_index_in_batch: the second index of leaf node states in state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
            # e.g. the state of the leaf node in (x, y) is state_batch_in_search_path[x, y], where x is state_index, y is batch_index.
            # The index of value prefix hidden state of the leaf node are in the same manner.
            """
            MCTS stage 1: Selection
                Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
            """
            state_index_in_search_path, state_index_in_batch, last_actions = tree.batch_traverse(
                roots, self._pb_c_base, self._pb_c_init, self._discount_factor, min_max_stats_lst, results)

            # obtain the state for leaf node
            states = []
            for ix, iy in zip(state_index_in_search_path, state_index_in_batch):
                states.append(state_batch_in_search_path[ix][iy])
            states = self._env.from_state_list(states)
            last_actions = np.array(last_actions)

            """
            MCTS stage 2: Expansion
                At the final time-step l of the simulation, the state and reward/value_prefix are computed by the dynamics function.
                Then we calculate the policy_logits and value for the leaf node (state) by the prediction function. (aka. evaluation)
            """
            states, obs, all_legal_actions, n_legal_actions, rewards, dones = self._env.step(states, last_actions)
            
            logits, pred_values = self._predict_fn(obs)

            values = pred_values.reshape(-1)
            values = np.where(dones, rewards, values)

            state_batch_in_search_path.append(states)

            tree.batch_expand(
                i_sim, dones, rewards, logits, all_legal_actions, n_legal_actions, results)

            """
            MCTS stage 3: Backup
                At the end of the simulation, the statistics along the trajectory are updated.
            """
            # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
            # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
            # statistics.
            tree.batch_backpropagate(
                self._discount_factor, values, min_max_stats_lst, results)

        all_probs, all_values, all_actions = self._predict(
            roots, init_legal_actions_list, temperature, sample)
        if return_roots:
            return all_probs, all_values, all_actions, roots
        return all_probs, all_values, all_actions

    def _predict(
        self,
        roots: tree.Roots,
        legal_actions_list: List[List[int]],
        temperature: float = 1.0,
        sample: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the target action probabilities, values for learning and actions
        for acting from the roots after search.

        Parameters
        ----------
        roots : Roots
            The roots after search.
        legal_actions_list : List[List[int]], shape (n_legals_1,), (n_legals_2,), ..., (n_legals_n,)
            The list of legal actions for each state.
        temperature : float, default: 1.0
            The temperature used to adjust the sampling distribution.
        sample : bool, default: False
            Whether to sample action for acting. If False, select the argmax result.

        Returns
        -------
        probs : Tuple[np.ndarray], shape (parallel_episodes, action_dim1), (parallel_episodes, action_dim2), ...
            The target action probabilities of the roots for learning.
        values : np.ndarray, shape (parallel_episodes,)
            The target Q values of the roots.
        action : np.ndarray, shape (parallel_episodes, n_actions)
            The selected action of the roots for acting.
        """
        action_dim = self._env.action_space.n
        batch_size = roots.num
        # list: (batch_size, n_legal_actions)
        roots_visit_counts = roots.get_distributions()
        roots_values = roots.get_values()  # list: (batch_size,)

        all_probs = np.zeros((batch_size, action_dim), dtype=np.float32)
        all_actions = np.zeros(batch_size, dtype=np.int32)
        all_values = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            visit_counts, value = roots_visit_counts[i], roots_values[i]
            visit_counts = np.array(visit_counts)
            action_index_in_legal_action_set = select_action(
                visit_counts, temperature=temperature, deterministic=not sample)
            # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
            # entire action set.
            legal_actions = legal_actions_list[i]
            action = legal_actions[action_index_in_legal_action_set]

            probs = visit_counts / sum(visit_counts)

            all_probs[i, legal_actions] = probs
            all_actions[i] = action
            all_values[i] = value
        return all_probs, all_values, all_actions

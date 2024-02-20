from abc import ABC, abstractmethod
import warnings
from typing import Dict, Tuple, Union, NamedTuple, List, Any, Optional

import numpy as np
import numba
import torch as th
from gymnasium import spaces

import psutil


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


TensorDict = Dict[str, th.Tensor]


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray):
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.dones[batch_inds, env_indices].reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


dtype_dict = {
    np.bool_       : th.bool,
    np.uint8       : th.uint8,
    np.int8        : th.int8,
    np.int16       : th.int16,
    np.int32       : th.int32,
    np.int64       : th.int64,
    np.float16     : th.float16,
    np.float32     : th.float32,
    np.float64     : th.float64,
}


@numba.njit
def nstep_return_selfplay(rewards, to_play, gamma):
    returns = np.zeros_like(rewards)
    R0 = rewards[-1]
    R1 = -rewards[-1]
    returns[-1] = R0
    pl = to_play[-1]
    for step in np.arange(len(rewards) - 2, -1, -1):
        if to_play[step] == pl:
            R0 = gamma * R0 + rewards[step]
            returns[step] = R0
        else:
            R1 = gamma * R1 - rewards[step]
            returns[step] = R1
    return returns


@numba.njit
def nstep_return(rewards, gamma):
    returns = np.zeros_like(rewards)
    R = 0.0
    for step in np.arange(len(rewards) - 1, -1, -1):
        R = rewards[step] + gamma * R
        returns[step] = R
    return returns


class DMCBuffer:
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
        device: Union[th.device, str] = "auto",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.n_envs = n_envs
        self.pos = 0
        self.full = False
        self.start = np.zeros((self.n_envs,), dtype=np.int32)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.device = get_device(device)
        
        self.observations = th.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=dtype_dict[observation_space.dtype.type], device=self.device)
        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=dtype_dict[action_space.dtype.type], device=self.device)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: np.ndarray,
    ) -> None:
        batch_size = reward.shape[0]
        self.observations[self.pos] = obs
        self.actions[self.pos] = action.reshape((batch_size, self.action_dim))
        self.rewards[self.pos] = np.array(reward)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def mark_episode(self, env_ind, gamma):
        start = self.start[env_ind]
        pos = self.pos
        if pos <= start:
            end = pos + self.buffer_size
            batch_inds = np.arange(start, end) % self.buffer_size
        else:
            batch_inds = np.arange(start, pos)
        self.start[env_ind] = pos

        returns = nstep_return(self.rewards[batch_inds, env_ind], gamma)
        self.returns[batch_inds, env_ind] = th.from_numpy(returns).to(self.device)

    def get_data_indices(self):
        if not self.full:
            indices = np.arange(self.start.min())
        else:
            # if np.all(pos >= self.start):
            #     indices = np.arange(pos, self.start.min() + self.buffer_size) % self.buffer_size
            # elif np.all(pos < self.start):
            #     indices = np.arange(pos, self.start.min())
            # else:
            start = self.pos
            end = np.where(self.pos >= self.start, self.start + self.buffer_size, self.start).min()
            indices = np.arange(start, end) % self.buffer_size
        return indices

    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds, :, :].reshape(-1, *self.obs_shape),
            self.actions[batch_inds, :, :].reshape(-1, self.action_dim),
            self.returns[batch_inds, :].reshape(-1),
        )
        return data


def create_obs(observation_space: spaces.Dict, shape: Tuple[int, ...], device: Union[th.device, str] = "cpu"):
    obs_shape = get_obs_shape(observation_space)
    obs = {
        key: th.zeros(
            (*shape, *_obs_shape),
            dtype=dtype_dict[observation_space[key].dtype.type], device=device)
        for key, _obs_shape in obs_shape.items()
    }
    return obs

class DMCDictBuffer:
    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        selfplay: bool = False,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]
        self.selfplay = selfplay

        self.action_dim = get_action_dim(action_space)
        self.n_envs = n_envs
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.start = np.zeros((self.n_envs,), dtype=np.int32)

        self.buffer_size = max(buffer_size // n_envs, 1)
        
        self.observations = {
            key: th.zeros(
                (self.buffer_size, self.n_envs, *_obs_shape),
                dtype=dtype_dict[observation_space[key].dtype.type], device=self.device)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.actions = th.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=dtype_dict[action_space.dtype.type], device=self.device)
        self.returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if self.selfplay:
            self.to_play = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        self._observations = {
            key: th.zeros(
                (self.buffer_size * self.n_envs, *_obs_shape),
                dtype=dtype_dict[observation_space[key].dtype.type], device=self.device)
            for key, _obs_shape in self.obs_shape.items()
        }
        self._actions = th.zeros(
            (self.buffer_size * self.n_envs, self.action_dim), dtype=dtype_dict[action_space.dtype.type], device=self.device)
        self._returns = th.zeros((self.buffer_size * self.n_envs,), dtype=th.float32, device=self.device)

        obs_nbytes = 0
        for _, obs in self.observations.items():
            obs_nbytes += obs.nbytes

        total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.returns.nbytes
        total_memory_usage = total_memory_usage / 1e9 * 2
        print(f"Total gpu memory usage: {total_memory_usage:.2f}GB")

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: np.ndarray,
        to_play: Optional[np.ndarray] = None,
    ) -> None:
        batch_size = reward.shape[0]
        for key in self.observations.keys():
            self.observations[key][self.pos] = obs[key]
        self.actions[self.pos] = action.reshape((batch_size, self.action_dim))
        self.rewards[self.pos] = np.array(reward)
        if self.selfplay:
            self.to_play[self.pos] = to_play

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def mark_episode(self, env_ind, gamma):
        start = self.start[env_ind]
        pos = self.pos
        if pos <= start:
            end = pos + self.buffer_size
            batch_inds = np.arange(start, end) % self.buffer_size
        else:
            batch_inds = np.arange(start, pos)
        self.start[env_ind] = pos

        if self.selfplay:
            returns = nstep_return_selfplay(self.rewards[batch_inds, env_ind], self.to_play[batch_inds, env_ind], gamma)
        else:
            returns = nstep_return(self.rewards[batch_inds, env_ind], gamma)
        self.returns[batch_inds, env_ind] = th.from_numpy(returns).to(self.device)

    def get_data_indices(self):
        if not self.full:
            indices = np.arange(self.start.min())
            # print(0, self.start.min(), self.pos, self.start, self.full)
        else:
            start = self.pos
            end = np.where(self.pos >= self.start, self.start + self.buffer_size, self.start).min()
            indices = np.arange(start, end) % self.buffer_size
            # print(start, end, self.pos, self.start, self.full)
        return indices

    def _get_samples(self, batch_inds: np.ndarray):
        l = len(batch_inds) * self.n_envs
        for key, obs in self.observations.items():
            _obs = self._observations[key]
            _obs[:l, :] = obs[batch_inds, :, :].flatten(0, 1)
        self._actions[:l, :] = self.actions[batch_inds, :, :].reshape(-1, self.action_dim)
        self._returns[:l] = self.returns[batch_inds, :].reshape(-1)

        data = (
            {key: _obs[:l] for key, _obs in self._observations.items()},
            self._actions[:l],
            self._returns[:l],
        )
        return data


class DictReplayBuffer:
    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]
    observations: Dict[str, np.ndarray]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        mem_available = psutil.virtual_memory().available

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        obs_nbytes = 0
        for _, obs in self.observations.items():
            obs_nbytes += obs.nbytes

        total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

        if total_memory_usage > mem_available:
            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            warnings.warn(
                "This system does not have apparently enough memory to store the complete "
                f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos


    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :return:
        """
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs_ = {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}

        assert isinstance(obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            dones=self.to_torch(self.dones[batch_inds, env_indices]).reshape(-1, 1),
            rewards=self.to_torch(self.rewards[batch_inds, env_indices].reshape(-1, 1)),
        )

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

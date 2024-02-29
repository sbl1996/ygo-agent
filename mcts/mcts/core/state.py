from typing import Tuple, Sequence

import numpy as np

class State:

    def __init__(self, batch_shape: Tuple[int, ...], store):
        assert isinstance(store, np.ndarray)
        self.store = store

        self.batch_shape = batch_shape
        self.ndim = len(batch_shape)

    def get_state_keys(self):
        return self.store

    @classmethod
    def from_item(cls, item):
        return cls((1,), np.array([item], dtype=np.int32))

    def reshape(self, batch_shape: Tuple[int, ...]):
        self.batch_shape = batch_shape
        self.ndim = len(batch_shape)
        return self

    def item(self):
        assert self.ndim == 1 and self.batch_shape[0] == 1
        return self.store[0]

    @classmethod
    def from_state_list(cls, state_list, batch_shape=None):
        if isinstance(state_list[0], State):
            batch_shape_ = (len(state_list),)
        elif isinstance(state_list[0], Sequence):
            batch_shape_ = (len(state_list), len(state_list[0]))
            assert isinstance(state_list[0][0], State)
        else:
            raise ValueError("Invalid dim of states")
        if batch_shape is None:
            batch_shape = batch_shape_
        else:
            assert len(batch_shape) == 2 and len(batch_shape_) == 1
        if len(batch_shape) == 2:
            states = [s for ss in state_list for s in ss]
        else:
            states = state_list
        state_keys = np.concatenate([s.store for s in states], dtype=np.int32, axis=0)
        return State(batch_shape, state_keys)

    def _get_by_index(self, batch_shape, indices):
        state_keys = self.store[indices]
        return State(batch_shape, state_keys)

    def __getitem__(self, item):
        return self.get(item)

    def get(self, i):
        if self.ndim == 2:
            assert isinstance(i, tuple)
            i = i[0] * self.batch_shape[1] + i[1]
        i = np.array([i], dtype=np.int32)
        return self._get_by_index((1,), i)

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return f'State(batch_shape={self.batch_shape}, ndim={self.ndim})'

    def __str__(self) -> str:
        return f'State(batch_shape={self.batch_shape}, ndim={self.ndim})'

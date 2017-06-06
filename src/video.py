# -*- coding: utf-8 -*-


import numpy as np


class Buffer():

    @property
    def original(self) -> np.ndarray:
        return self._original

    @property
    def binary(self) -> np.ndarray:
        return self._binary

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def framecount(self) -> int:
        return self._original.shape[0]

    def __iter__(self):
        for i in range(self.framecount):
            yield i, self.original[i]

    def __init__(self, original: np.ndarray):
        assert original.ndim == 4
        assert original.size > 0

        n, w, h, _ = original.shape

        self._original = original
        self._binary = np.zeros((n, w, h))
        self._edges = np.zeros((n, w, h))

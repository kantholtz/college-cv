# -*- coding: utf-8 -*-


from datetime import datetime

import numpy as np
import scipy.ndimage as scnd
from . import logger

log = logger(__name__)


# ---


def norm(arr):
    # fmt = '%s (%s) - min: %d, max: %d'
    # print(fmt % ('normalizing: ', arr.shape, np.min(arr), np.max(arr)))

    a_min = np.min(arr)
    if a_min < 0:
        arr += abs(a_min)

    a_max = np.max(arr)
    if a_max == 0:
        return arr.astype(np.uint8)

    norm = (arr / a_max) * 255
    # print(fmt % ('result: ', arr.shape, np.min(norm), np.max(norm)))
    return norm.astype(np.uint8)


# ---


class Pipeline():

    def __getitem__(self, key):
        if type(key) is int:
            return self._modules_executed[key]

        if type(key) is str:
            return self._modules[key]

        raise TypeError

    def __add__(self, mod):
        log.info('adding mod "%s" to pipeline', mod.name)

        mod.pipeline = self
        self._module_names.append(mod.name)
        self._modules[mod.name] = mod

        return self

    # --- initialization

    def __init__(self, arr):
        self._modules_executed = []
        self._modules = {}
        self._module_names = []

        # set up initial module
        initial = Module(self)
        initial.arr = arr
        self._modules_executed.append(initial)

    def run(self):
        t_start = datetime.now()
        log.info('>>> running pipeline')

        for name in self._module_names:
            log.debug('executing module %s', name)
            mod = self._modules[name]
            mod.arr = mod.execute()
            self._modules_executed.append(mod)

        log.info('>>> finished pipeline in %sms',
                 (datetime.now() - t_start).microseconds / 1000)


class Module():

    @property
    def name(self) -> str:
        return self._name

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pl: Pipeline):
        self._pipeline = pl

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, arr):
        assert 0 <= np.amin(arr) and np.amax(arr) <= 255
        self._arr = arr

    def __init__(self, name: str):
        self._name = name

    def execute(self) -> None:
        raise NotImplementedError

# ---


class Binarize(Module):

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        assert 0 <= threshold and threshold <= 1
        self._threshold = threshold

    def __init__(self, name: str):
        super().__init__(name)
        self._threshold = 0.3

    def execute(self) -> np.ndarray:
        log.info('binarize: using a threshold of %f' % self.threshold)
        src = self.pipeline[-1].arr.astype(np.int64)
        tgt = src[:, :, 0] - (src[:, :, 1] + src[:, :, 2])

        e = 255 * self.threshold
        tgt[tgt < e] = 0
        tgt[tgt > e] = 255

        return tgt


class Morph(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        src = self.pipeline[-1].arr
        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src)] = 255
        return tgt

# -*- coding: utf-8 -*-


# import math
from datetime import datetime

import numpy as np
# import skimage.draw as skd
import scipy.ndimage as scnd
# import skimage.transform as skt

from . import logger
log = logger(__name__)


# ---


def _norm(arr):
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


# _sin = {a: math.sin(math.radians(a)) for a in range(180)}
# _cos = {a: math.cos(math.radians(a)) for a in range(180)}


# def _hess2point(h, w):

#     h_bound = h - 1
#     w_bound = w - 1

#     # max_rad = int(round(math.sqrt(h**2 + w**2) + .5))

#     def in_img(p):
#         y, x = p
#         return 0 <= y and y < h and 0 <= x and x < w

#     def get(a, r):
#         # print('get points: a=%d, r=%d' % (a, r))
#         if a == 0:
#             off = min(r, w_bound)
#             return (0, off), (h_bound, off)

#         if a == 90:
#             off = min(r, h_bound)
#             return (off, 0), (off, w_bound)

#         # if a > 90:
#         #     r = -max_rad + r

#         p_lt = r / _sin[a], 0
#         p_rt = (-w_bound * _cos[a] + r) / _sin[a], w_bound
#         p_up = 0, r / _cos[a]
#         p_dn = h_bound, (-h_bound * _sin[a] + r) / _cos[a]

#         points = (tuple(int(round(t)) for t in p)
#                   for p in (p_lt, p_rt, p_up, p_dn))

#         # print('a=%d, r=%d, cos=%f, sin=%f' % (
#         #     a, r, self.cos[a], sin[a]))

#         # print('pre filter: %s' % str(tuple(points)))
#         return tuple(filter(in_img, points))

#     return get


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
        self._mod_initial = Module('_')
        self._mod_initial.arr = arr

    def run(self):
        t_start = datetime.now()
        log.info('>>> running pipeline')

        self._modules_executed = [self._mod_initial]
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
        assert type(name) is str
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

    @property
    def amplification(self):
        return self._amplification

    @amplification.setter
    def amplification(self, amplification: float):
        assert amplification > 0
        self._amplification = amplification

    def __init__(self, name: str):
        super().__init__(name)
        self._threshold = 0.3
        self._amplification = 1

    def execute(self) -> np.ndarray:
        log.info('binarize: threshold=%f, amplification=%f',
                 self.threshold, self.amplification)

        a = self.amplification
        src = self.pipeline[-1].arr.astype(np.int64)
        tgt = (src[:, :, 0] * a) - (src[:, :, 1] + src[:, :, 2])

        e = 255 * self.threshold
        tgt[tgt < e] = 0
        tgt[tgt > e] = 255

        return tgt


class Dilate(Module):

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        assert type(iterations) is int
        self._iterations = iterations

    def __init__(self, name: str):
        super().__init__(name)
        self._iterations = 2

    def execute(self) -> np.ndarray:
        log.info('dilate with %d iterations', self.iterations)
        src = self.pipeline[-1].arr
        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src, iterations=self.iterations)] = 255
        return tgt


class Fill(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        src = self.pipeline[-1].arr
        labeled, labels = scnd.label(np.invert(src.astype(np.bool)))

        max_count = 0
        max_label = -1

        for i in range(labels):
            count = np.count_nonzero(labeled == i)
            if count > max_count:
                max_count = count
                max_label = i

        tgt = np.copy(src)
        tgt[labeled == max_label] = 255
        return tgt


class Edger(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        log.info('edgering')
        src = self.pipeline[-1].arr
        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src)] = 255
        return _norm(src.astype(np.bool) ^ tgt.astype(np.bool))


class Hough(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        log.info('detecting lines via hough transformation')

        # src = self.pipeline[-1].arr
        tgt = self.pipeline[0].arr / 3

        # _, angles, dists = skt.hough_line_peaks(
        #     *skt.hough_line(src),
        #     min_angle=30,
        #     min_distance=10)

        # for a, r in zip(np.rad2deg(angles), dists):
        #     log.info('looking at a=%s, r=%s', str(a), str(r))

        # lines = skt.probabilistic_hough_line(src, threshold=0.5)
        # tgt = np.copy(self.pipeline[0].arr)

        # for points in [np.asarray(l)[:, ::-1] for l in lines]:
        #     tgt[skd.line(*np.ravel(points))] = 1

        return tgt

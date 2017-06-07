# -*- coding: utf-8 -*-


import numpy as np

from . import pipeline as pl


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
        self._binary = np.zeros((n, w, h), dtype=np.uint8)
        self._edges = np.zeros((n, w, h), dtype=np.uint8)


class Pipeline():

    @property
    def binary(self) -> bool:
        return self._binary

    @property
    def edges(self) -> bool:
        return self._edges

    def __init__(self, config=None, binary=False, edges=False):
        self._binary = binary
        self._edges = edges

        if config is None:
            self._conf = {
                'ReferenceColor': '200, 20, 20',
                'Threshold': '80',
                'Dilate': '3',
                'Erode': '3'}
        else:
            config = config['options']

        # step I : preprocessing
        self._mod_binarize = pl.Binarize('binarize')
        self._mod_binarize.threshold = int(config['Threshold'])

        clr = config['ReferenceColor'].split(',')
        self._mod_binarize.reference_color = tuple(map(int, clr))

        self._mod_dilate = pl.Dilate('dilate')
        self._mod_dilate.iterations = int(config['Dilate'])

        self._mod_erode = pl.Erode('erode')
        self._mod_erode.iterations = int(config['Erode'])

        # step II : edge detection
        self._mod_fill = pl.Fill('fill')
        self._mod_edger = pl.Edger('edger')

        # step III : line and sign detection
        self._mod_hough = pl.Hough('hough')

    # ---

    def binarize(self, src):
        return self._mod_erode.apply(
            self._mod_dilate.apply(
                self._mod_binarize.apply(src)))

    def edge(self, src):
        return self._mod_edger.apply(
            self._mod_fill.apply(src))

    def detect(self, src, reference):
        self._mod_hough.binarized = reference
        self._mod_hough.apply(src)

        return (self._mod_hough.barycenter.items(),
                self._mod_hough.pois)


class ROI():

    @property
    def r0(self) -> int:
        return self._r0

    @property
    def r1(self) -> int:
        return self._r1

    @property
    def c0(self) -> int:
        return self._c0

    @property
    def c1(self) -> int:
        return self._c1

    @property
    def vy(self) -> np.array:
        return self._vy

    @property
    def vx(self) -> np.array:
        return self._vx

    @property
    def dead(self) -> bool:
        return self.health <= 0

    @property
    def health(self) -> int:
        return self._health

    # ---

    def __init__(self,
                 h: int, w: int,              # source dimension
                 vy: np.array, vx: np.array,  # vertices
                 lifespan: int):              # for life decay

        self._vy = vy
        self._vx = vx
        self._health = lifespan

        vy_min = np.min(vy)
        vy_max = np.max(vy)

        vx_min = np.min(vx)
        vx_max = np.max(vx)

        ry_off = vy_max - vy_min
        rx_off = vx_max - vx_min

        self._r0 = vy_min - ry_off
        self._r0 = 0 if self.r0 < 0 else self.r0

        self._r1 = vy_max + ry_off
        self._r1 = h - 1 if self.r1 >= h else self.r1

        self._c0 = vx_min - rx_off
        self._c0 = 0 if self.c0 < 0 else self.c0

        self._c1 = vx_max + rx_off
        self._c1 = w - 1 if self.c1 >= w else self.c1

    def punish(self):
        self._health -= 1

    def intersects(self, other) -> bool:
        return not (
            self.r0 > other.r1 or
            self.r1 < other.r0 or
            self.c0 > other.c1 or
            self.c1 < other.c0)

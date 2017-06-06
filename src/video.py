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
        self._binary = np.zeros((n, w, h))
        self._edges = np.zeros((n, w, h))


class Pipeline():

    def __init__(self, config=None):
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

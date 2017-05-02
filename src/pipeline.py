# -*- coding: utf-8 -*-


from datetime import datetime

import numpy as np
import PyQt5.QtWidgets as qtw

from . import gui
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

    @property
    def main_module(self) -> gui.ImageModule:
        return self._main_module

    @property
    def tab_widget(self) -> qtw.QTabWidget:
        return self._tab_widget

    @property
    def modules(self) -> dict:
        return self._modules

    # --- initialization

    def __init__(self, main_module, tab_widget):
        self._main_module = main_module
        self._tab_widget = tab_widget
        self._modules_ordered = []
        self._modules = {}

    def add_module(self, name, module) -> None:
        self._modules_ordered.append(name)
        self._modules[name] = module

    def run(self):
        t_start = datetime.now()
        log.info('>>> running pipeline')

        for mod in self._modules_ordered:
            log.debug('executing module %s', mod)
            self.modules[mod].execute()

        log.info('>>> finished pipeline in %sms',
                 (datetime.now() - t_start).microseconds / 1000)


class Module():

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def execute(self) -> None:
        raise NotImplementedError

# ---


class Select(Module):

    def __init__(self, pipeline):
        super().__init__(pipeline)

    def execute(self) -> None:
        log.info('select: exposing red color')
        src = self.pipeline.main_module.view.image.arr.astype(np.int64)

        tgt = src[:, :, 0] - (src[:, :, 1] + src[:, :, 2])
        tgt[tgt < 0] = 0

        self.arr = tgt

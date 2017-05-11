# -*- coding: utf-8 -*-

"""

Maintains the processing pipeline and handles displaying processing steps

"""


import numpy as np
import PyQt5.QtWidgets as qtw

from . import logger
from . import gui_image
from . import pipeline as pl

log = logger(name=__name__)

# def _build_pipeline(self, module) -> None:
#     log.info('initializing pipeline')
#     pl = pipeline.Pipeline(module.view.image.arr)

#     pl + pipeline.Binarize('binarize')
#     pl + pipeline.Morph('morph')

#     pl.run()

#     log.info('drawing results')
#     mod_binarized = gui_image.ImageModule(pl['binarize'].arr)
#     mod_binarized.add_view(pl['morph'].arr, stats_right=True)
#     self._tab_widget.addTab(mod_binarized, 'Binarized')


class PipelineGUI():

    @property
    def main(self):
        return self._main

    @property
    def tab_widget(self) -> qtw.QTabWidget:
        return self._tab_widget

    @property
    def pipeline(self) -> pl.Pipeline:
        return self._pipeline

    @property
    def tabs(self) -> list:
        return self._tabs[:]

    def __add__(self, tab):
        assert isinstance(tab, Tab)
        log.info('adding tab "%s"', tab.name)
        self._tabs.append(tab)

    def __iter__(self):
        return self.tabs.__iter__()

    # ---

    def _init_gui(self, arr):
        module = gui_image.ImageModule(arr)

        layout = qtw.QVBoxLayout()
        layout.addWidget(module, stretch=1)

        origin = qtw.QWidget()
        origin.setLayout(layout)

        self.tab_widget.addTab(origin, 'Source')

    def __init__(self, main, tab_widget: qtw.QTabWidget, arr: np.ndarray):
        self._main = main
        self._tab_widget = tab_widget
        self._tabs = []

        self._init_gui(arr)

        # add tabs, initialize pipeline
        self._pipeline = pl.Pipeline(arr)
        self + Preprocessing()

        # fire in the hole
        for tab in self:
            for mod in tab:
                self.pipeline + mod

        self._initialized = False
        self.update()

    # ---

    def update(self) -> None:
        self.pipeline.run()

        for tab in self:
            tab.update()

        if not self._initialized:
            for tab in self:
                self.tab_widget.addTab(tab.widget, tab.name)
            self._initialized = True


class Tab():

    @property
    def name(self) -> str:
        return self._name

    @property
    def widget(self) -> qtw.QWidget:
        return self._widget

    def __add__(self, mod):
        assert isinstance(mod, pl.Module)
        log.info('adding mod "%s" to tab "%s"', mod.name, self.name)
        self._mods.append(mod)

    def __iter__(self):
        return self._mods.__iter__()

    def __init__(self, name: str):
        self._name = name
        self._mods = []

    # interface

    def update(self):
        raise NotImplementedError

# ---


class Preprocessing(Tab):

    def _init_gui(self):
        self._widget = gui_image.ImageModule(self._mod_binarize.arr)
        self._view_morph = self.widget.add_view(
            self._mod_morph.arr, stats_right=True)

    def __init__(self):
        super().__init__('Preprocessing')

        self._mod_binarize = pl.Binarize('binarize')
        self._mod_morph = pl.Morph('morph')

        for mod in [self._mod_binarize, self._mod_morph]:
            self + mod

    def update(self):
        try:
            self.widget.view.image.arr = self._mod_binarize.arr
            self._view_morph = self._mod_morph.arr

        except AttributeError:
            self._init_gui()

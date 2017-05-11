# -*- coding: utf-8 -*-

"""

Maintains the processing pipeline and handles displaying processing steps

"""


import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw

from . import logger
from . import gui_image
from . import pipeline as pl

log = logger(name=__name__)


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
        self + Hough()

        # fire in the hole
        for tab in self:
            tab.ping = self.update
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


class Tab(qtw.QWidget):

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
        super().__init__()
        self._name = name
        self._mods = []

    # interface

    def ping(self):
        """
        Re-run the processing pipeline.
        This method is overwritten by PipelineGUI.

        """
        raise NotImplementedError

    def update(self):
        """
        To update the views of the tab with recently computed images

        """
        raise NotImplementedError

# ---


class Preprocessing(Tab):

    BIN_SLIDER_MIN = 1
    BIN_SLIDER_MAX = 100
    BIN_SLIDER_FAC = 100

    MORPH_SLIDER_MIN = 1
    MORPH_SLIDER_MAX = 10

    def _update_bin_threshold(self):
        val = self._slider_bin.value() / Preprocessing.BIN_SLIDER_FAC
        self._mod_binarize.threshold = val
        self.ping()

    def _update_morph_iterations(self):
        val = self._slider_morph.value()
        self._mod_morph.iterations = val
        self.ping()

    def _init_gui(self):
        self._widget = gui_image.ImageModule(self._mod_binarize.arr)
        self._view_morph = self.widget.add_view(self._mod_morph.arr)
        self._view_fill = self.widget.add_view(self._mod_fill.arr)

        # ---

        controls = self.widget.view.controls
        layout = qtw.QVBoxLayout()

        # ---

        layout.addWidget(qtw.QLabel('Binarization threshold'))
        slider = self._slider_sobel = qtw.QSlider(qtc.Qt.Horizontal, self)
        slider.setFocusPolicy(qtc.Qt.NoFocus)

        slider.setMinimum(Preprocessing.BIN_SLIDER_MIN)
        slider.setMaximum(Preprocessing.BIN_SLIDER_MAX)

        val = self._mod_binarize.threshold * Preprocessing.BIN_SLIDER_FAC
        slider.setValue(val)

        slider.sliderReleased.connect(self._update_bin_threshold)
        layout.addWidget(slider)
        self._slider_bin = slider

        # ---

        layout.addWidget(qtw.QLabel('Dilation iterations'))
        slider = self._slider_sobel = qtw.QSlider(qtc.Qt.Horizontal, self)
        slider.setFocusPolicy(qtc.Qt.NoFocus)

        slider.setMinimum(Preprocessing.MORPH_SLIDER_MIN)
        slider.setMaximum(Preprocessing.MORPH_SLIDER_MAX)

        val = self._mod_morph.iterations
        slider.setValue(val)

        slider.sliderReleased.connect(self._update_morph_iterations)
        layout.addWidget(slider)
        self._slider_morph = slider

        # ---

        controls.addLayout(layout)

    def __init__(self):
        super().__init__('Preprocessing')

        self._mod_binarize = pl.Binarize('binarize')
        self._mod_morph = pl.Morph('morph')
        self._mod_fill = pl.Fill('fill')

        for mod in [self._mod_binarize, self._mod_morph, self._mod_fill]:
            self + mod

    def update(self):
        try:
            self.widget.view.image.arr = self._mod_binarize.arr
            self._view_morph.image.arr = self._mod_morph.arr
            self._view_fill.image.arr = self._mod_fill.arr

        except AttributeError:
            self._init_gui()


class Hough(Tab):

    def _init_gui(self):
        self._widget = gui_image.ImageModule(self._mod_hough.arr)

    def __init__(self):
        super().__init__('Hough Selection')
        self._mod_hough = pl.Hough('hough')
        for mod in [self._mod_hough]:
            self + mod

    def update(self):
        try:
            self.widget.view.image.arr = self._mod_hough.arr

        except AttributeError:
            self._init_gui()

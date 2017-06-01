# -*- coding: utf-8 -*-

"""

Maintains the processing pipeline and handles displaying processing steps

"""


import numpy as np
import skimage.draw as skd
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
        self._widget = gui_image.ImageModule(arr)

        layout = qtw.QVBoxLayout()
        layout.addWidget(self._widget, stretch=1)

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
        self + EdgeDetection()
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

        try:
            self._view_result.image.arr = self._tabs[-1].result
        except AttributeError:
            self._view_result = self._widget.add_view(self._tabs[-1].result)


class Tab(qtw.QWidget):

    @property
    def name(self) -> str:
        return self._name

    @property
    def widget(self) -> qtw.QWidget:
        return self._widget

    # --- utility

    def _add_slider(self,
                    parent: qtw.QLayout,
                    callback,
                    smin: int,
                    smax: int,
                    sfac: float = 1.0,
                    initial: float = None,
                    label: str = None):
        """
        Utility method to easily add sliders

        """
        qlabel = None
        slider = qtw.QSlider(qtc.Qt.Horizontal, self)
        slider.setFocusPolicy(qtc.Qt.NoFocus)

        slider.setMinimum(smin)
        slider.setMaximum(smax)

        if initial is not None:
            slider.setValue(initial * sfac)

        # proxies the slider-released-event to update the label
        # containing the current value and applying the factor
        # before invoking the provided callback
        def proxy():
            val = slider.value() / sfac
            if qlabel is not None:
                qlabel.setText(str(val))
            callback(val)

        slider.sliderReleased.connect(proxy)

        if label is not None:
            layout = qtw.QHBoxLayout()
            layout.addWidget(qtw.QLabel(label))

            qlabel = qtw.QLabel(str(initial or ""))
            layout.addWidget(qlabel)

            parent.addLayout(layout)

        parent.addWidget(slider)

    def _mod_proxy(self, mod, prop):
        def _callback(val):
            mod.__setattr__(prop, val)
            self.ping()
        return _callback

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

    def _init_gui(self):
        self._widget = gui_image.ImageModule(self._mod_binarize.arr)
        self._view_dilate = self.widget.add_view(
            self._mod_dilate.arr, stats_right=True)

        # ---

        controls = self.widget.view.controls
        layout = qtw.QVBoxLayout()

        init = self._mod_binarize.amplification
        fn = self._mod_proxy(self._mod_binarize, 'amplification')
        self._add_slider(layout, fn, 5, 10, 5,
                         initial=init, label='Red amplification')

        init = self._mod_binarize.threshold
        fn = self._mod_proxy(self._mod_binarize, 'threshold')
        self._add_slider(layout, fn, 1, 100, 100,
                         initial=init, label='Binarization δ')

        init = self._mod_dilate.iterations
        fn = self._mod_proxy(self._mod_dilate, 'iterations')
        self._add_slider(layout, fn, 1, 10,
                         initial=init, label='Dilation iterations')

        controls.addLayout(layout)

    def __init__(self):
        super().__init__('Binarization and Dilation')

        self._mod_binarize = pl.Binarize('binarize')
        self._mod_dilate = pl.Dilate('dilate')

        self + self._mod_binarize
        self + self._mod_dilate

    def update(self):
        try:
            self.widget.view.image.arr = self._mod_binarize.arr
            self._view_dilate.image.arr = self._mod_dilate.arr

        except AttributeError:
            self._init_gui()


class EdgeDetection(Tab):

    def _init_gui(self):
        self._widget = gui_image.ImageModule(self._mod_edger.arr)
        # self._widget = gui_image.ImageModule(self._mod_fill.arr)
        # self._view_edger = self.widget.add_view(
        #     self._mod_edger.arr, stats_right=True)

    def __init__(self):
        super().__init__('Edge Exposure')
        # self._mod_fill = pl.Fill('fill')
        self._mod_edger = pl.Edger('edger')

        # self + self._mod_fill
        self + self._mod_edger

    def update(self):
        try:
            self.widget.view.image.arr = self._mod_edger.arr
            # self.widget.view.image.arr = self._mod_fill.arr
            # self._view_edger.image.arr = self._mod_edger.arr

        except AttributeError:
            self._init_gui()


class Hough(Tab):

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, barycenter):
        tgt = self._mod_hough.arr / 3

        for y, x, r in barycenter:
            rr, cc, vv = skd.circle_perimeter_aa(y, x, r, shape=tgt.shape)
            tgt[rr, cc, 0] += vv * 255
            tgt[rr, cc, 1] += vv * 255
            tgt[rr, cc, 2] += vv * 255
            tgt[tgt > 255] = 255

            rr, cc = skd.circle(y, x, r, shape=tgt.shape)
            tgt[rr, cc] *= 3

        self._result = tgt

    def _init_gui(self, arr: np.ndarray):
        self._widget = gui_image.ImageModule(arr)

        # ---

        controls = self.widget.view.controls
        layout = qtw.QVBoxLayout()

        init = self._mod_hough.min_angle
        fn = self._mod_proxy(self._mod_hough, 'min_angle')
        self._add_slider(layout, fn, 1, 90,
                         initial=init, label='Minimum angle')

        init = self._mod_hough.min_distance
        fn = self._mod_proxy(self._mod_hough, 'min_distance')
        self._add_slider(layout, fn, 1, 200,
                         initial=init, label='Minimum distance')

        controls.addLayout(layout)

    def _unpack(self, dic):
        """
        Takes a mapping int -> int -> X and returns [X]

        """
        vals = [d.values() for d in dic.values()]
        return [a for b in vals for a in b]

    def __init__(self):
        super().__init__('Hough Selection')
        self._mod_hough = pl.Hough('hough')
        self + self._mod_hough

    def update(self):
        tgt = self._mod_hough.arr / 4
        h, w, _ = tgt.shape

        def _bound(val: int, bound: int) -> int:
            if val < 0:
                return 0
            elif val > bound:
                return bound
            else:
                return val

        for a, d in zip(self._mod_hough.angles, self._mod_hough.dists):
            # TODO division by zero for a=[01]

            y0 = int(d / np.sin(a))
            y1 = int((d - w * np.cos(a)) / np.sin(a))

            x0 = int(d / np.cos(a))
            x1 = int((d - h * np.sin(a)) / np.cos(a))

            y0, y1 = [int(_bound(y, h-1)) for y in (y0, y1)]
            x0, x1 = [int(_bound(x, w-1)) for x in (x0, x1)]

            # TODO: revise; see pipeline.py Hough.execute
            if a > 0:
                y0, y1 = y1, y0

            if (y0 >= 0 and y1 >= 0):
                rr, cc, vv = skd.line_aa(y0, x0, y1, x1)
                tgt[rr, cc, 0] += vv * 255
                tgt[tgt > 255] = 255

        # ---
        for y, x in self._unpack(self._mod_hough.pois):
            rr, cc = skd.circle(y, x, 3)
            tgt[rr, cc] = [255, 255, 255]

        for y, x, r in self._mod_hough.barycenter.values():
            rr, cc = skd.circle(y, x, 3)
            tgt[rr, cc] = [0, 125, 255]

            rr, cc, vv = skd.circle_perimeter_aa(y, x, r, shape=tgt.shape)
            tgt[rr, cc, 0] += vv * 255

            tgt[tgt > 255] = 255

        self.result = self._mod_hough.barycenter.values()

        try:
            # self._result = tgt
            self.widget.view.image.arr = tgt

        except AttributeError:
            self._init_gui(tgt)

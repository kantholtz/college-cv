# -*- coding: utf-8 -*-

"""

Collection of GUI elements

"""


import numpy as np
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

import PyQt5.QtWidgets as qtw

from . import image


# --


class BarGraph(qtw.QWidget):
    """
    Widget to display a histogram in form of a bar graph.
    The data provided corresponds to the format of numpy histogram calls.

    """

    @property
    def size(self) -> qtc.QSize:
        return self._size

    @size.setter
    def size(self, size: (int, int)) -> None:
        h, w = size
        self._size = qtc.QSize(w, h)

    @property
    def color(self) -> qtg.QColor:
        return self._color

    @color.setter
    def color(self, color):
        assert len(color) == 3
        self._color = qtg.QColor(*tuple(color))

    def __init__(
            self,
            dims: (int, int),
            data: np.ndarray,
            color=(66, 66, 66)):
        super().__init__()
        self.color = color
        self.updateData(dims, data)

    def updateData(self, dims: (int, int), data) -> None:
        h, w = dims
        self.size = dims
        self.norm = h - (data / np.amax(data)) * h
        self.update()

    def sizeHint(self) -> qtc.QSize:
        return self.size

    def paintEvent(self, e: qtg.QPaintEvent) -> None:
        qp = qtg.QPainter()
        qp.begin(self)

        geom = self.geometry()
        h, w = geom.height(), geom.width()

        # draw background
        clr = qtg.QColor(170, 170, 170)
        brush = qtg.QBrush(clr, qtc.Qt.Dense6Pattern)
        qp.setBrush(brush)
        qp.drawRect(0, 0, w-1, h-1)

        # draw graph
        step = w / len(self.norm)
        pth = qtg.QPainterPath()

        curr_x = 0
        pth.moveTo(curr_x, h-1)

        for val in self.norm:
            pth.lineTo(curr_x, val)
            curr_x += step
            pth.lineTo(curr_x, val)

        pth.lineTo(w-1, h-1)
        pth.lineTo(0, h-1)

        brush = qtg.QBrush(self.color, qtc.Qt.SolidPattern)
        qp.setBrush(brush)
        qp.drawPath(pth)

        qp.end()


class ImageView(qtw.QWidget):
    """
    Base class for displaying both color and grayscale images.
    Optionally, different statistical informations are shown.

    """

    @property
    def image(self) -> image.Image:
        return self._image

    @property
    def controls(self) -> qtw.QBoxLayout:
        return self._controls

    # ---

    def _add_widget(
            self,
            parent: qtw.QWidget,
            widget: qtw.QWidget,
            **dargs) -> qtw.QWidget:
        parent.addWidget(widget, **dargs)
        return widget

    def _init_stats_layout(self) -> qtw.QLayout:
        layout = qtw.QVBoxLayout()

        self._init_stats(layout)

        spacer = qtw.QLabel()
        spacer.setMargin(10)
        layout.addWidget(spacer)

        self._controls = qtw.QVBoxLayout()
        layout.addLayout(self.controls)

        layout.addStretch(1)
        return layout

    def _init_layout(self, right: bool) -> None:
        layout = qtw.QHBoxLayout()
        stats_layout = self._init_stats_layout()

        if not right and right is not None:
            layout.addLayout(stats_layout)

        layout.addWidget(self.image, stretch=1)

        if right:
            layout.addLayout(stats_layout)

        self.setLayout(layout)

    def __init__(self, arr: np.ndarray, stats_right: bool):
        assert type(stats_right) is bool or stats_right is None
        super().__init__()

        self._image = image.Image(arr)
        self._init_layout(stats_right)

    # interface

    def _init_stats(self, layout) -> qtw.QBoxLayout:
        raise NotImplemented


class ColorImageView(ImageView):
    """
    For displaying statistical informations about color images.

    """

    def _update_text_stats(self) -> None:
        h, w = self.image.shape
        self._ql_height.setText('Height: %dpx' % h)
        self._ql_width.setText('Width: %dpx' % w)

    def _update_histograms(self) -> None:
        self._qw_hist_r.updateData(*self._init_histogram(0))
        self._qw_hist_g.updateData(*self._init_histogram(1))
        self._qw_hist_b.updateData(*self._init_histogram(2))

    def _init_histogram(self, channel: int) -> ((int, int), [float]):
        view = self.image.arr[:, :, channel]
        counts, _ = np.histogram(view, bins=256, range=(0, 256))
        return (100, 256), (counts)

    def _init_text_stats(self) -> qtw.QLayout:
        layout = qtw.QVBoxLayout()

        self._ql_height = self._add_widget(layout, qtw.QLabel(''))
        self._ql_width = self._add_widget(layout, qtw.QLabel(''))

        layout.addStretch(1)
        return layout

    def _init_stats(self, layout) -> qtw.QBoxLayout:

        # histograms
        self._qw_hist_r = self._add_widget(
            layout,
            BarGraph(*self._init_histogram(0), color=(150, 0, 0)))

        self._qw_hist_g = self._add_widget(
            layout,
            BarGraph(*self._init_histogram(1), color=(0, 150, 0)))

        self._qw_hist_b = self._add_widget(
            layout,
            BarGraph(*self._init_histogram(2), color=(0, 0, 150)))

        # text stats
        layout.addLayout(self._init_text_stats())
        self._update_text_stats()

    def __init__(self, arr: np.ndarray, stats_right=False):
        super().__init__(arr, stats_right)


class GrayscaleImageView(ImageView):
    """
    For displaying statistical informations about grayscale images.

    """
    def _update_text_stats(self) -> None:
        arr = self.image.arr
        h, w = self.image.shape

        self._ql_height.setText('Height: %dpx' % h)
        self._ql_width.setText('Width: %dpx' % w)
        self._ql_min.setText('Minimum gray value: %d' % np.min(arr))
        self._ql_max.setText('Maximum gray value: %d' % np.max(arr))
        self._ql_avg.setText('Average gray value: %d' % np.average(arr))

    def _init_text_stats(self) -> qtw.QLayout:
        layout = qtw.QVBoxLayout()

        self._ql_height = self._add_widget(layout, qtw.QLabel(''))
        self._ql_width = self._add_widget(layout, qtw.QLabel(''))
        self._ql_min = self._add_widget(layout, qtw.QLabel(''))
        self._ql_max = self._add_widget(layout, qtw.QLabel(''))
        self._ql_avg = self._add_widget(layout, qtw.QLabel(''))

        layout.addStretch(1)
        return layout

    def _update_histograms(self) -> None:
        self._qw_hist.updateData(*self._init_histogram())
        self._qw_hist_cum.updateData(*self._init_histogram(cum=True))

    def _init_histogram(self, cum=False) -> ((int, int), [float]):
        counts, _ = np.histogram(self.image.arr, bins=256, range=(0, 256))
        return (100, 256), (np.cumsum(counts) if cum else counts)

    def _init_stats(self, layout: qtw.QLayout) -> None:

        # histograms

        self._qw_hist = self._add_widget(
            layout,
            BarGraph(*self._init_histogram()))

        self._qw_hist_cum = self._add_widget(
            layout,
            BarGraph(*self._init_histogram(cum=True)))

        # text stats

        layout.addLayout(self._init_text_stats())
        self._update_text_stats()

    def __init__(self, arr: np.ndarray, stats_right=False):
        super().__init__(arr, stats_right)


class ImageModule(qtw.QWidget):
    """
    Displays ImageViews with common zoom and offset.

    """

    ZOOM_FACTOR = 100.

    @property
    def view(self) -> image.Image:
        return self.views[0]

    @property
    def views(self) -> [image.Image]:
        return self._views

    @property
    def zoom(self) -> float:
        return self.view.image.zoom

    @zoom.setter
    def zoom(self, zoom):
        for view in self.views:
            view.image.zoom = zoom

    @property
    def offset(self) -> (int, int):
        self.view.image.offset

    @offset.setter
    def offset(self, offset):
        for view in self.views:
            view.image.offset = offset

    @property
    def controls(self) -> qtw.QBoxLayout:
        return self._controls

    @property
    def mouse_position(self) -> (int, int):
        self.view.image.mouse_position

    @mouse_position.setter
    def mouse_position(self, pos):
        for view in self.views:
            view.image.mouse_position = pos

    # ---

    def handle_tracking(self, data):
        cpos, ipos, value = data

        if cpos is not None:
            self._ql_cpos.setText('%d, %d' % cpos)
            self.mouse_position = data[0]
        else:
            self._ql_cpos.setText('gone')

        if ipos is not None:
            self._ql_ipos.setText('%d, %d' % ipos)
        else:
            self._ql_ipos.setText('gone')

        try:
            if value is not None:
                if len(value) == 1:
                    self._ql_value.setText('Gray %d' % value[0])
                elif len(value) == 3:
                    self._ql_value.setText('RGB %d %d %d' % tuple(value))
            else:
                self._ql_value.setText('nothing')
        except TypeError:
            print('ignore value of tracking', type(value), value)

    def handle_offset(self, pos):
        for view in self.views:
            view.image.offset = pos

    # ---

    def _add_widget(self, parent, widget, **dargs):
        parent.addWidget(widget, **dargs)
        return widget

    def _init_controls(self) -> qtw.QLayout:
        """

        Initializes the zoom slider and position information

        :returns: the hbox layout
        :rtype: qtw.QHBoxLayout

        """
        layout = qtw.QHBoxLayout()
        left = qtw.QHBoxLayout()

        left.addWidget(qtw.QLabel('Zoom:'))

        # zoom/slider

        self._zoom_slider = qtw.QSlider(qtc.Qt.Horizontal, self)
        self._zoom_slider.setFocusPolicy(qtc.Qt.NoFocus)
        factor = ImageModule.ZOOM_FACTOR

        self._zoom_slider.setMinimum(factor / 4.)
        self._zoom_slider.setMaximum(factor * 5.)
        self._zoom_slider.setValue(factor)

        self._zoom_slider.valueChanged[int].connect(self.zoom)
        left.addWidget(self._zoom_slider)

        # zoom indicator

        self._zoom_label = qtw.QLabel()
        left.addWidget(self._zoom_label)

        # zoom/reset

        zoom_reset = qtw.QPushButton("reset zoom", self)

        def reset(_):
            for view in self.views:
                view.image.zoom = 1.

            self._zoom_slider.setValue(factor)
            view.image.update()

        zoom_reset.clicked.connect(reset)
        left.addWidget(zoom_reset)

        # position

        right = qtw.QHBoxLayout()

        right.addStretch(1)
        self._add_widget(right, qtw.QLabel('Position on canvas: '))
        self._ql_cpos = self._add_widget(right, qtw.QLabel('gone'))
        self._add_widget(right, qtw.QLabel('| Position in image:'))
        self._ql_ipos = self._add_widget(right, qtw.QLabel('gone'))
        self._add_widget(right, qtw.QLabel('| Value:'))
        self._ql_value = self._add_widget(right, qtw.QLabel('nothing'))

        # compose

        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=1)

        return layout

    def _init_layout(self):
        layout = qtw.QVBoxLayout()

        self._viewbox = qtw.QHBoxLayout()
        layout.addLayout(self._viewbox, stretch=1)
        layout.addLayout(self._init_controls())

        self._controls = qtw.QVBoxLayout()
        layout.addLayout(self.controls)
        self.setLayout(layout)

    def __init__(self, arr, stats_right=False):
        super().__init__()

        self._views = []

        self._init_layout()
        self.add_view(arr, stats_right=stats_right)

    # ---

    def add_view(self, arr, stats_right=None):
        """
        Adds a new ImageView with optional stats.

        """
        if len(arr.shape) == 2:
            view = GrayscaleImageView(arr, stats_right=stats_right)
        elif len(arr.shape) == 3:
            view = ColorImageView(arr, stats_right=stats_right)
        else:
            raise Exception('Unsupported image shape')

        self.views.append(view)
        view.image.sig_tracking.connect(self.handle_tracking)
        view.image.sig_offset.connect(self.handle_offset)

        self._viewbox.addWidget(view)
        self.update()

        return view

    def remove_view(self, view):
        self.views.remove(view)
        view.deleteLater()
        self.update()

    # alter canvas

    def zoom(self, factor):
        """
        Zoom all images by the provided factor

        :param factor: some float value
        :returns: nothing
        :rtype: None

        """
        zoom = float(factor) / ImageModule.ZOOM_FACTOR
        for view in self.views:
            view.image.zoom = zoom

        # self.zoomIndicator.setText('%3d%%' % factor)

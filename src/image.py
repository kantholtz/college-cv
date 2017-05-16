# -*- coding: utf-8 -*-


from datetime import datetime

import numpy as np
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import qimage2ndarray as qim2nd

from . import logger

log = logger(name=__name__)


class Image(qtw.QWidget):
    """

    Base class for painting images that can be zoomed and panned on a canvas

    """

    @property
    def arr(self) -> np.ndarray:
        return self._arr

    @arr.setter
    def arr(self, arr):
        assert type(arr) is np.ndarray
        self._arr = arr
        self.update()

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, zoom):
        assert type(zoom) is float
        self._zoom = zoom
        self.update()

    @property
    def offset(self) -> (int, int):
        return self._offset

    @offset.setter
    def offset(self, offset):
        assert len(offset) == 2
        assert all(type(o) is int for o in offset)
        self._offset = offset
        self.update()

    @property
    def mouse_position(self) -> (int, int):
        return self._mouse_position

    @mouse_position.setter
    def mouse_position(self, pos):
        assert len(pos) == 2
        assert all(type(p) is int for p in pos)
        self._mouse_position = pos
        self.update()

    @property
    def dim(self) -> (int, int):
        """
        Returns the dimensions of the QWidget geometry (height, width).

        """
        return self.geometry().height(), self.geometry().width()

    @property
    def shape(self) -> (int, int):
        """
        Returns (height, width) of the image regardless
        of color channel amount.

        """
        return self.arr.shape[0], self.arr.shape[1]

    def sizeHint(self):
        h, w = self.shape
        return qtc.QSize(w, h + 20)

    # --- events

    sig_offset = qtc.pyqtSignal(tuple)
    sig_tracking = qtc.pyqtSignal(tuple)
    sig_hover = qtc.pyqtSignal(bool)

    def enterEvent(self, evt):
        evt.accept()
        self.sig_hover.emit(True)

    def leaveEvent(self, evt):
        evt.accept()
        self.sig_hover.emit(False)
        self.sig_tracking.emit((None, None, None))

    def mouseMoveEvent(self, evt):
        evt.accept()
        pos = int(evt.localPos().y()), int(evt.localPos().x())

        # tracking calculation
        def calc_ipos(coord):
            offset, current = coord
            pos = -offset + current
            return int(pos // self.zoom)

        ipos = tuple(map(calc_ipos, zip(self.offset, pos)))
        if (
                ipos[0] < 0 or
                ipos[1] < 0 or
                ipos[0] >= self.arr.shape[0] or
                ipos[1] >= self.arr.shape[1]):

            ipos = None
            value = None
        else:
            value = self.arr[ipos]
            if len(self.arr.shape) == 2:
                value = [value]

        self.sig_tracking.emit((pos, ipos, value))

        # panning
        def pan(coord):
            current, start, previous = coord
            return int(previous + current - start)

        if self._mouse_start is not None:
            new = map(pan, zip(pos, self._mouse_start, self._offset_initial))
            self.sig_offset.emit(tuple(new))

    def mousePressEvent(self, evt):
        evt.accept()
        pos = evt.y(), evt.x()
        self._mouse_start = pos
        self._offset_initial = self.offset

    def mouseReleaseEvent(self, evt):
        evt.accept()
        self._mouse_start = None

    # --- initialization

    def _qimfac(self, cropped: np.ndarray) -> qtg.QImage:
        """
        Returns an QImage instance based on a cropped ndarray

        """
        if len(cropped.shape) == 2:
            return qim2nd.gray2qimage(cropped)
        elif len(cropped.shape) == 3:
            return qim2nd.array2qimage(cropped)

        raise Exception('Unsupported image shape')

    def __init__(self, arr):
        """
        Create a new image, provide an array

        """
        super().__init__()
        self.setMouseTracking(True)

        self._arr = arr
        self._zoom = 1
        self._offset = (10, 10)
        self._mouse_position = None
        self._mouse_start = None

    # --- painting

    def paintEvent(self, e):
        """
        Called by qt's event loop for redrawing

        """
        t_start = datetime.now()

        dim_h, dim_w = self.dim
        img_h, img_w = self.arr.shape[0], self.arr.shape[1]
        off_y, off_x = self.offset

        # t_start = datetime.now()
        # initialize painter

        qp = qtg.QPainter()
        qp.begin(self)

        # background
        clr = qtg.QColor(170, 170, 170)
        qp.setBrush(qtg.QBrush(clr, qtc.Qt.BDiagPattern))
        qp.drawRect(0, 0, dim_w-1, dim_h-1)

        # paint
        ih, iw = self.arr.shape[0], self.arr.shape[1]
        qim = self._qimfac(self.arr).scaled(
            self.zoom * iw, self.zoom * ih,
            qtc.Qt.KeepAspectRatio,
            qtc.Qt.SmoothTransformation)

        qp.drawImage(
            max(0, off_x), max(0, off_y),    # top left point of paint device
            qim,
            -min(0, off_x), -min(0, off_y),  # top left point of image device
            self.zoom * img_w, self.zoom * img_h)

        # hit marker
        if self.mouse_position is not None:
            qp.setPen(qtg.QColor(255, 0, 0))

            mouse_y, mouse_x = self.mouse_position
            qp.drawPoint(mouse_x, mouse_y)

            a, b, x, y = 10, 6, mouse_x, mouse_y

            perms = ((x+a, y, x+b, y),
                     (x, y+a, x, y+b),
                     (x-a, y, x-b, y),
                     (x, y-a, x, y-b))

            for perm in perms:
                qp.drawLine(*perm)

            qp.setPen(qtg.QColor(0, 0, 0))

        # _ts('time for drawing: %s\n', t_start)
        qp.setBrush(qtc.Qt.NoBrush)
        qp.drawRect(0, 0, dim_w-1, dim_h-1)
        qp.end()

        log.log(log.TRACE, 'repainting took %sμs',
                (datetime.now() - t_start).microseconds)

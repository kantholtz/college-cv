#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

GUI based application for finding yield signs.

"""

import sys
import argparse

import numpy as np
import skimage.io as skio
import PyQt5.QtWidgets as qtw

from src import logger
from src import gui_pipeline

log = logger(name=__name__[2:-2])


class MainWindow(qtw.QMainWindow):

    @property
    def pipeline(self) -> gui_pipeline.PipelineGUI:
        return self._pipeline

    def _action(self, text, handler, tip=None, shortcut=None):
        action = qtw.QAction(text, self)
        action.triggered.connect(handler)

        if tip is not None:
            action.setStatusTip(tip)

        if shortcut is not None:
            action.setShortcut(shortcut)

        return action

    def _create_menu(self, parent, text, actions):
        menu = parent.addMenu(text)

        for action in actions:
            menu.addAction(action)

        return menu

    def _mods(self):
        mods = []
        for tab in self.pipeline:
            mods += [mod for mod in tab]
        return mods

    # --- handler

    def _handle_load_file(self):
        fname, _ = qtw.QFileDialog.getOpenFileName(
            self, 'Open bitmap')

        if fname:
            self.load_file(fname)

    def _handle_save_file(self):
        log.info('saving images')

        i = 0
        for mod in self._mods()[:-1]:
            fname = 'out.%d.%s.png' % (i, mod.name)
            skio.imsave(fname, mod.arr.astype(np.uint8))
            i += 1

        tab = self.pipeline.tabs[-1]
        skio.imsave('out.%d.hough.png' % i, tab.widget.view.image.arr)

        i += 1
        skio.imsave('out.%d.result.png' % i, tab.result)
        log.info('done saving images')

    # --- initialization

    def _init_file_menu(self, menu: qtw.QMenuBar) -> None:
        actions = [
            self._action(
                'Load &File',
                self._handle_load_file,
                shortcut='Ctrl+f'),
            self._action(
                '&Save all images',
                self._handle_save_file,
                shortcut='Ctrl+s'),
            self._action(
                '&Exit',
                qtw.qApp.quit,
                shortcut='Ctrl+q')]
        self._create_menu(menu, 'File', actions)

    def _init_menu(self) -> None:
        menu = self.menuBar()
        self._init_file_menu(menu)

    def _init_main(self, arr: np.ndarray) -> None:
        tabs = qtw.QTabWidget()
        self.setCentralWidget(tabs)
        self._pipeline = gui_pipeline.PipelineGUI(self, tabs, arr)
        # tabs.setCurrentIndex(1)

    def __init__(self, app: qtw.QApplication, fname=None):
        super().__init__()
        self.setMouseTracking(True)

        # set size
        scr_geom = app.desktop().screenGeometry()
        scr_height, scr_width = scr_geom.height(), scr_geom.width()
        self.resize(scr_width * .8, scr_height * .8)
        self.center()

        # initialize
        self._init_menu()
        self.setWindowTitle('Detect me some yield signs')

        if fname is not None:
            self.load_file(fname)

        self.message('Ready')
        self.show()

    # --- utility

    def center(self) -> None:
        qr = self.frameGeometry()
        cp = qtw.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def message(self, msg: str, time=None) -> None:
        """
        Display a message in the status bar

        :param msg: Message
        :param time: Optional timeout in ms

        """
        bar = self.statusBar()
        if time is None:
            bar.showMessage(msg)
        else:
            bar.showMessage(msg, time)

    def load_file(self, fname: str) -> None:
        log.info('opening %s', fname)
        arr = skio.imread(fname)
        self._init_main(arr)
        self.message('Finished loading bitmap', 2000)

#
# --- initialization
#


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fname',
        nargs=1, type=str,
        help='open a file directly')

    return parser.parse_args()


def main(args):
    fname = None if args.fname is None else args.fname[0]

    # ---

    log.info('starting the application')
    app = qtw.QApplication(sys.argv)
    win = MainWindow(app, fname=fname) # noqa  # pylint: disable=unused-import
    sys.exit(app.exec_())


if __name__ == '__main__':
    args = parse_args()
    main(args)

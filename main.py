#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

GUI based application for finding yield signs.

"""

import sys
import argparse

import PyQt5.QtWidgets as qtw


class MainWindow(qtw.QMainWindow):

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

    # --- handler

    def _handle_load_file(self):
        print('load file')

    # --- initialization

    def _init_file_menu(self, menu: qtw.QMenuBar) -> None:
        actions = [
            self._action(
                'Load &File',
                self._handle_load_file,
                shortcut='Ctrl+f'),
            self._action(
                '&Exit',
                qtw.qApp.quit,
                shortcut='Ctrl+q')]
        self._create_menu(menu, 'File', actions)

    def _init_menu(self) -> None:
        menu = self.menuBar()
        self._init_file_menu(menu)

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
    app = qtw.QApplication(sys.argv)
    win = MainWindow(app, fname=fname) # noqa  # pylint: disable=unused-import
    sys.exit(app.exec_())


if __name__ == '__main__':
    args = parse_args()
    main(args)

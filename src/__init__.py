# -*- coding: utf-8 -*-

import logging
import logging.config

from datetime import datetime


logging.addLevelName(5, 'TRACE')
logging.config.fileConfig('logging.conf')


def logger(name):
    logger = logging.getLogger(name)
    logger.TRACE = logging.getLevelName('TRACE')
    return logger


# ---


def tmeasure(l, msg: str):
    tstart = datetime.now()

    def _done(*args):
        tdelta = (datetime.now() - tstart).microseconds / 1000
        l(msg, *((tdelta, ) + args))

    return _done

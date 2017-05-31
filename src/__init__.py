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


def tmeasure(l: logging.Logger, msg: str, *args):
    tstart = datetime.now()

    def _done():
        tdelta = (datetime.now() - tstart).microseconds / 1000
        l.info(msg, *((tdelta, ) + args))

    return _done

# -*- coding: utf-8 -*-

import logging
import logging.config


logging.addLevelName(5, 'TRACE')
logging.config.fileConfig('logging.conf')


def logger(name):
    logger = logging.getLogger(name)
    logger.TRACE = logging.getLevelName('TRACE')
    return logger

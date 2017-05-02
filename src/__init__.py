# -*- coding: utf-8 -*-

import logging
import logging.config


logging.config.fileConfig('logging.conf')


def logger(name):
    return logging.getLogger(name)

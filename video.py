#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script to detect yield signs from video.  Videos are loaded into ram,
so you need to split the file or else it starts swapping ;)

"""


import sys
import argparse


import numpy as np
import skvideo.io as skvio


from src import pipeline as pl
from src import logger
from src import tmeasure
log = logger(name=__name__[2:-2])


# ---

def load(fname: str) -> np.ndarray:
    vid = skvio.vread(fname)
    n, h, w, depth = vid.shape
    if depth != 3:
        print('You need to provide a colored video!')
        sys.exit(2)

    log.info('loaded video with %d frames and resulution %d, %d',
             n, h, w)

    return vid


def save(fname: str, vid: np.ndarray):
    log.info('saving to %s', fname)
    skvio.vwrite(fname, vid)


def start(vid: np.ndarray):
    log.info('initializing modules')

    mod_binarize = pl.Binarize('binarize')
    mod_binarize.threshold = 120

    log.info('start processing')
    n, h, w, _ = vid.shape

    binary = np.zeros((n, h, w))

    done = tmeasure(log.info, 'took %sms')
    for frame in range(n):
        binary[frame] = mod_binarize.apply(vid[frame].astype(np.int64))
        sys.stdout.write('.')
        sys.stdout.flush()

    print('')
    done()

    return binary


#
# --- initialization
#


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'f_in', type=str,
        help='input file')

    parser.add_argument(
        'f_out', type=str,
        help='output file')

    return parser.parse_args()


def main(args):
    log.info('starting the application')
    save(args.f_out, start(load(args.f_in)))


if __name__ == '__main__':
    args = parse_args()
    main(args)

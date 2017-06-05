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
import skimage.draw as skd


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

    # preprocessing

    mod_binarize = pl.Binarize('binarize')
    mod_binarize.threshold = 90
    mod_binarize.reference_color = (150, 20, 20)

    mod_dilate = pl.Dilate('dilate')
    mod_dilate.iterations = 4

    mod_erode = pl.Erode('erode')
    mod_erode.iterations = 4

    # edging

    mod_fill = pl.Fill('fill')
    mod_edger = pl.Edger('edger')

    # detecting

    mod_hough = pl.Hough('hough')

    # start

    log.info('start processing')
    n, h, w, _ = vid.shape
    binary = np.zeros((n, h, w))

    done = tmeasure(log.info, 'took %sms')
    for frame in range(n):

        binary[frame] = mod_binarize.apply(vid[frame].astype(np.int64))
        binary[frame] = mod_dilate.apply(binary[frame])
        binary[frame] = mod_erode.apply(binary[frame])

        binary[frame] = mod_fill.apply(binary[frame])
        binary[frame] = mod_edger.apply(binary[frame])

        vid[frame] = vid[frame] / 3

        mod_hough.binarized = binary[frame]
        mod_hough.apply(binary[frame])
        for (p0, p1, p2), (y, x) in mod_hough.barycenter.items():
            pois = (p0, p1), (p0, p2), (p1, p2)
            py, px = zip(*[mod_hough.pois[a][b] for a, b in pois])

            rr, cc = skd.polygon_perimeter(
                py + (py[0], ),
                px + (px[0], ),
                shape=vid[frame].shape)

            vid[frame, rr, cc] = [255, 255, 255]

            r = (np.max(py) - np.min(py)) * 2
            rr, cc = skd.circle_perimeter(
                y, x, r, shape=vid[frame].shape)

            vid[frame, rr, cc, 0] = 255

        sys.stdout.write('.')
        sys.stdout.flush()

    print('')
    done()

    return vid


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

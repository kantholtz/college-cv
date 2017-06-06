#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script to detect yield signs from video.  Videos are loaded into ram,
so you need to split the file or else it starts swapping ;)

"""


import sys
import argparse
import configparser

import numpy as np
import skvideo.io as skvio
import skimage.draw as skd
from tqdm import tqdm


from src import pipeline as pl
from src import logger
from src import tmeasure


# ---


log = logger(name=__name__[2:-2])
cfg = configparser.ConfigParser()


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


def _initialize(config):
    if config is None:
        config = {
            'ReferenceColor': '200, 20, 20',
            'Dilate': '3',
            'Erode': '3'}
    else:
        config = config['options']

    # step I : preprocessing
    mod_binarize = pl.Binarize('binarize')
    mod_binarize.threshold = 50

    clr = config['ReferenceColor'].split(',')
    mod_binarize.reference_color = tuple(map(int, clr))

    mod_dilate = pl.Dilate('dilate')
    mod_dilate.iterations = int(config['Dilate'])

    mod_erode = pl.Erode('erode')
    mod_erode.iterations = int(config['Erode'])

    def segments(src: np.ndarray) -> np.ndarray:
        return mod_erode.apply(mod_dilate.apply(mod_binarize.apply(src)))

    # step II : edge detection
    mod_fill = pl.Fill('fill')
    mod_edger = pl.Edger('edger')

    def edges(src: np.ndarray) -> np.ndarray:
        return mod_edger.apply(mod_fill.apply(src))

    # step III : line and sign detection
    mod_hough = pl.Hough('hough')

    return segments, edges, mod_hough


def _draw_indicator(vid, frame, mod_hough):
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


def process(vid: np.ndarray, config=None, binary=False, edges=False):
    log.info('initializing modules')
    if edges:
        binary = True

    fn_segments, fn_edges, mod_hough = _initialize(config)

    # start

    log.info('start processing')
    n, h, w, _ = vid.shape
    vid_binary = np.zeros((n, h, w))
    vid_edges = np.zeros((n, h, w))

    print('')
    done = tmeasure(log.info, 'took %sms')
    for frame in tqdm(range(n)):

        vid_binary[frame] = fn_segments(vid[frame].astype(np.int64))
        if binary and not edges:
            continue

        vid_edges[frame] = fn_edges(vid_binary[frame])
        if edges:
            continue

        vid[frame] = vid[frame] / 3

        mod_hough.binarized = vid_binary[frame]
        mod_hough.apply(vid_edges[frame])
        _draw_indicator(vid, frame, mod_hough)

    print('')
    done()

    if edges:
        return vid_edges
    elif binary:
        return vid_binary
    else:
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

    parser.add_argument(
        '--config', type=str, nargs=1,
        help='configuration file')

    parser.add_argument(
        '--binary',
        action='store_true',
        help='only apply segmentation and morphology')

    parser.add_argument(
        '--edges',
        action='store_true',
        help='only apply --binary and edge detection')

    args = parser.parse_args()

    if args.binary and args.edges:
        print('either provide --edges or --binary')
        parser.print_help()
        sys.exit(2)

    return args


def main(args):
    log.info('starting the application')

    if len(args.config) > 0:
        cfg.read(args.config[0])
        if 'options' not in cfg:
            print('You need to provide an [option] section')
            sys.exit(2)

    result = process(load(args.f_in),
                     config=cfg if args.config else None,
                     binary=args.binary,
                     edges=args.edges)

    save(args.f_out, result)


if __name__ == '__main__':
    args = parse_args()
    main(args)

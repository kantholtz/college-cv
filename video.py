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


from src import video
from src import logger
from src import tmeasure
from src import pipeline as pl

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


def _draw_indicator(frame, y, x, vy, vx):
    rr, cc = skd.polygon_perimeter(
        vy + (vy[0], ),
        vx + (vx[0], ),
        shape=frame.shape)

    frame[rr, cc] = [255, 255, 255]

    r = (np.max(vy) - np.min(vy)) * 2
    rr, cc = skd.circle_perimeter(
        y, x, r, shape=frame.shape)

    frame[rr, cc, 0] = 255


def _draw_roi(frame, roi):
    ry, rx = roi

    # clockwise from topleft
    rr, cc = skd.polygon_perimeter(
        (ry[0], ry[0], ry[1], ry[1], ry[0]),
        (rx[0], rx[1], rx[1], rx[0], rx[0]))

    frame[rr, cc] = [255, 255, 255]


def _calc_roi(h, w, vy, vx):
    vy_min = np.min(vy)
    vy_max = np.max(vy)

    vx_min = np.min(vx)
    vx_max = np.max(vx)

    ry_off = (vy_max - vy_min) / 2
    rx_off = (vx_max - vx_min) / 2

    ry_min = vy_min - ry_off
    ry_min = 0 if ry_min < 0 else ry_min

    ry_max = vy_max + ry_off
    ry_max = h - 1 if ry_max >= h else ry_max

    rx_min = vx_min - rx_off
    rx_min = 0 if rx_min < 0 else rx_min

    rx_max = vx_max + rx_off
    rx_max = w - 1 if rx_max >= w else rx_max

    return (ry_min, ry_max), (rx_min, rx_max)


def _indicate(buf, frame, mod_hough, off_y, off_x):
    rois = []

    _, w, h, _ = buf.original.shape

    for (p0, p1, p2), (y, x) in mod_hough.barycenter.items():
        pois = (p0, p1), (p0, p2), (p1, p2)
        vy, vx = zip(*[mod_hough.pois[a][b] for a, b in pois])

        vy = tuple(map(lambda y: off_y + y, vy))
        vx = tuple(map(lambda x: off_x + x, vx))

        _draw_indicator(frame, y, x, vy, vx)
        roi = _calc_roi(w, h, vy, vx)

        _draw_roi(frame, roi)
        rois.append(roi)

    return rois


def _process(buf, frame, i, pipe, workload):
    fn_segments, fn_edges, mod_hough = pipe
    binary, edges = workload

    buf.binary[i] = fn_segments(frame.astype(np.int64))
    if binary and not edges:
        return []

    buf.edges[i] = fn_edges(buf.binary[i])
    if edges:
        return []

    frame //= 3

    mod_hough.binarized = buf.binary[i]
    mod_hough.apply(buf.edges[i])
    return _indicate(buf, frame, mod_hough, 0, 0)


def process(vid: np.ndarray, config=None, binary=False, edges=False):
    log.info('initializing modules')
    if edges:
        binary = True

    fn_segments, fn_edges, mod_hough = _initialize(config)

    log.info('start processing')

    n, w, h, _ = vid.shape
    buf = video.Buffer(vid)
    rois = []

    print('')
    done = tmeasure(log.info, 'took %sms')
    for i, frame in tqdm(buf, total=buf.framecount, unit='frames'):

        if len(rois) == 0:
            rois = [(0, w-1), (0, h-1)]

        # TODO refactor
        rois = _process(
            buf, frame, i,
            (fn_segments, fn_edges, mod_hough),
            (binary, edges))

    print('')
    done()

    if edges:
        return buf.edges
    elif binary:
        return buf.binary
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

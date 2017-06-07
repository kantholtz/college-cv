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


def _draw_roi(arr, i, roi, val):
    ry, rx = roi

    y0, y1 = ry[0] - 1, ry[1]
    x0, x1 = rx[0] - 1, rx[1]

    # clockwise from topleft
    rr, cc = skd.polygon_perimeter(
        (y0, y0, y1, y1, y0),
        (x0, x1, x1, x0, x0))

    arr[i, rr, cc] = val


def _calc_roi(h, w, vy, vx):
    vy_min = np.min(vy)
    vy_max = np.max(vy)

    vx_min = np.min(vx)
    vx_max = np.max(vx)

    ry_off = (vy_max - vy_min) // 2
    rx_off = (vx_max - vx_min) // 2

    ry_min = vy_min - ry_off
    ry_min = 0 if ry_min < 0 else ry_min

    ry_max = vy_max + ry_off
    ry_max = h - 1 if ry_max >= h else ry_max

    rx_min = vx_min - rx_off
    rx_min = 0 if rx_min < 0 else rx_min

    rx_max = vx_max + rx_off
    rx_max = w - 1 if rx_max >= w else rx_max

    return (ry_min, ry_max), (rx_min, rx_max)


def _get_vertices(keys, pois):
    p0, p1, p2 = keys
    vertices = (p0, p1), (p0, p2), (p1, p2)
    vy, vx = zip(*[pois[a][b] for a, b in vertices])
    return vy, vx


def _find_rois(buf, frame, barycenters, pois):
    _, w, h, _ = buf.original.shape
    rois = []

    for keys, (y, x) in barycenters:
        vy, vx = _get_vertices(keys, pois)
        _draw_indicator(frame, y, x, vy, vx)
        rois.append(_calc_roi(w, h, vy, vx))

    return rois


def _process(buf, frame, i, pipe, rois):
    _, w, h, _ = buf.original.shape

    # full scan
    if len(rois) == 0:
        buf.binary[i] = pipe.binarize(frame.astype(np.int64))
        if pipe.binary and not pipe.edges:
            return []

        buf.edges[i] = pipe.edge(buf.binary[i])
        if pipe.edges:
            return []

        frame //= 3
        barycenters, pois = pipe.detect(buf.edges[i], buf.binary[i])
        return _find_rois(buf, frame, barycenters, pois)

    # roi scan
    new_rois = []
    for roi in rois:
        (r0, r1), (c0, c1) = roi

        # it is not possible to have pipe.binary or pipe.edges and
        # running into this loop

        buf.binary[i, r0:r1, c0:c1] = pipe.binarize(
            frame[r0:r1, c0:c1].astype(np.int64))

        _draw_roi(buf.binary, i, roi, 255)

        buf.edges[i, r0:r1, c0:c1] = pipe.edge(
            buf.binary[i, r0:r1, c0:c1])

        _draw_roi(buf.edges, i, roi, 255)

        frame //= 3
        barycenters, pois = pipe.detect(
            buf.edges[i, r0:r1, c0:c1],
            buf.binary[i, r0:r1, c0:c1])

        # from skimage.io import imsave
        # imsave('original.png', buf.original[i, r0:r1, c0:c1])
        # imsave('binary.png', buf.binary[i, r0:r1, c0:c1])
        # imsave('edges.png', buf.edges[i, r0:r1, c0:c1])

        # print('\nabort!')
        # sys.exit(2)

        if len(barycenters) > 0:
            keys, (y, x) = list(barycenters)[0]
            vy, vx = _get_vertices(keys, pois)
            vy = tuple(map(lambda y: r0 + y, vy))
            vx = tuple(map(lambda x: c0 + x, vx))

            _draw_indicator(frame, r0+y, c0+x, vy, vx)
            new_rois.append(_calc_roi(w, h, vy, vx))

    return new_rois


def process(vid: np.ndarray, only, config=None) -> video.Buffer:
    only_binary, only_edges = only
    only_binary = only_edges or only_binary

    log.info('initializing modules')

    pipe = video.Pipeline(config, only_binary, only_edges)
    buf = video.Buffer(vid)

    log.info('start processing')

    n, w, h, _ = vid.shape
    rois = []

    print('')
    done = tmeasure(log.info, 'took %sms')
    stats = []

    for i, frame in tqdm(buf, total=buf.framecount, unit='frames'):
        stats.append('.' if len(rois) == 0 else 'x')
        rois = _process(buf, frame, i, pipe, rois)

    print(''.join(stats) + '\n')
    done()

    return buf

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

    parser.add_argument(
        '--save_all',
        action='store_true',
        help='save not only the result but all intermediate steps')

    args = parser.parse_args()

    if args.binary and args.edges:
        print('either provide --edges or --binary')
        parser.print_help()
        sys.exit(2)

    if args.save_all and (args.binary or args.edges):
        print('you can not save_all when using --edges or --binary')
        parser.print_help()
        sys.exit(2)

    return args


def _save(buf, args):
    if args.save_all:
        out = args.f_out.rsplit('.', maxsplit=1)

        out.insert(-1, 'binary')
        save('.'.join(out), buf.binary)

        out[-2] = 'edges'
        save('.'.join(out), buf.edges)

        save(args.f_out, buf.original)
        return

    if args.edges:
        save(args.f_out, buf.edges)
        return

    if args.binary:
        save(args.f_out, buf.binary)
        return

    save(args.f_out, buf.original)


def main(args):
    log.info('starting the application')

    if len(args.config) > 0:
        cfg.read(args.config[0])
        if 'options' not in cfg:
            print('You need to provide an [option] section')
            sys.exit(2)

    buf = process(load(args.f_in),
                  (args.binary, args.edges),
                  config=cfg if args.config else None)

    _save(buf, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# -*- coding: utf-8 -*-


from datetime import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np
import scipy.ndimage as scnd
import skimage.transform as skt

from . import logger
log = logger(__name__)


# ---


class Pipeline():

    def __getitem__(self, key):
        if type(key) is int:
            return self._modules_executed[key]

        if type(key) is str:
            return self._modules[key]

        raise TypeError

    def __add__(self, mod):
        log.info('adding mod "%s" to pipeline', mod.name)

        mod.pipeline = self
        self._module_names.append(mod.name)
        self._modules[mod.name] = mod

        return self

    def _ts(self, fmt: str, t_start: datetime, *args) -> None:
        delta = (datetime.now() - t_start).microseconds / 1000
        log.debug(fmt, delta, *args)

    def _run(self, name: str) -> None:
        mod = self._modules[name]
        mod.arr = mod.execute()
        self._modules_executed.append(mod)

    # --- initialization

    def __init__(self, arr):
        self._modules_executed = []
        self._modules = {}
        self._module_names = []

        # set up initial module
        self._mod_initial = Module('_')
        self._mod_initial.arr = arr

    def run(self):
        t_start = datetime.now()
        log.info('>>> running pipeline')

        self._modules_executed = [self._mod_initial]
        for name in self._module_names:
            log.debug('executing module %s', name)
            t_current_start = datetime.now()
            self._run(name)
            self._ts('execution took %sms', t_current_start)

        self._ts('finished pipeline in %sms', t_start)


class Module():

    @property
    def name(self) -> str:
        return self._name

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pl: Pipeline):
        self._pipeline = pl

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, arr):
        assert 0 <= np.amin(arr) and np.amax(arr) <= 255
        self._arr = arr

    def __init__(self, name: str):
        assert type(name) is str
        self._name = name

    def execute(self) -> None:
        raise NotImplementedError

# ---


class Binarize(Module):

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        assert 0 <= threshold and threshold <= 1
        self._threshold = threshold

    @property
    def amplification(self):
        return self._amplification

    @amplification.setter
    def amplification(self, amplification: float):
        assert amplification > 0
        self._amplification = amplification

    def __init__(self, name: str):
        super().__init__(name)
        self._threshold = 0.3
        self._amplification = 1.5

    def execute(self) -> np.ndarray:
        log.info('binarize: threshold=%f, amplification=%f',
                 self.threshold, self.amplification)

        a = self.amplification
        src = self.pipeline[-1].arr.astype(np.int64)
        tgt = (src[:, :, 0] * a) - (src[:, :, 1] + src[:, :, 2])

        e = 255 * self.threshold
        tgt[tgt < e] = 0
        tgt[tgt > e] = 255

        return tgt


class Dilate(Module):

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        self._iterations = int(iterations)

    def __init__(self, name: str):
        super().__init__(name)
        self._iterations = 2

    def execute(self) -> np.ndarray:
        log.info('dilate with %d iterations', self.iterations)
        src = self.pipeline[-1].arr
        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src, iterations=self.iterations)] = 255
        return tgt


class Fill(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        src = self.pipeline[-1].arr
        labeled, labels = scnd.label(np.invert(src.astype(np.bool)))

        max_count = 0
        max_label = -1

        for i in range(labels):
            count = np.count_nonzero(labeled == i)
            if count > max_count:
                max_count = count
                max_label = i

        tgt = np.copy(src)
        tgt[labeled == max_label] = 255
        return tgt


class Edger(Module):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        src = self.pipeline[-1].arr
        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src)] = 255
        return (src.astype(np.bool) ^ tgt.astype(np.bool)) * 255


class Hough(Module):

    @property
    def angles(self) -> np.array:
        return self._angles

    @property
    def dists(self) -> np.array:
        return self._dists

    @property
    def pois(self) -> (int, int):
        return self._pois

    @property
    def barycenter(self) -> dict:
        return self._barycenter

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        src = self.pipeline[-1].arr
        h, w = src.shape

        # --- apply hough transformation

        _, angles, dists = skt.hough_line_peaks(
            *skt.hough_line(src),
            min_angle=10,
            min_distance=100)

        self._angles = angles
        self._dists = dists

        # probabilistic hough: just for fun
        # lines = skt.probabilistic_hough_line(src, threshold=0.5)
        # tgt = np.copy(self.pipeline[0].arr)
        # for points in [np.asarray(l)[:, ::-1] for l in lines]:
        #     tgt[skd.line(*np.ravel(points))] = 255

        # --- detect points of intersection

        n = len(angles)

        # the returned angles span 180 degrees from -90 to 90
        # assuming the image coordinate [0, 0] starts north west
        # then the corresponding angles are:
        # [-1, 0] = -90
        # [ 0, 1] =   0
        # [ 1, 0] =  90

        # initialize normal vectors of the found lines
        # provided by the hough transformation
        hough_vecs = np.ndarray(shape=(2, n))
        hough_vecs[0] = np.sin(angles)  # y coordinates
        hough_vecs[1] = np.cos(angles)  # x coordinates

        # calculate the reference points based on the distance
        ref_points = hough_vecs * dists

        # calculate normal vectors describing the lines' direction
        hough_vecs_norm = np.ndarray(shape=(2, n))
        hough_vecs_norm[0] = hough_vecs[1]
        hough_vecs_norm[1] = -hough_vecs[0]

        # calculate points on each line for creating
        # the homogeneous coordinates
        line_points = ref_points + (hough_vecs_norm * 100)

        # create homogeneous coordinates
        ref_points = np.vstack((ref_points, np.ones(n)))
        line_points = np.vstack((line_points, np.ones(n)))
        h_coords = np.cross(ref_points.T, line_points.T)

        intersections = defaultdict(set)
        ipoints = defaultdict(dict)
        self._pois = set()

        # calculate points of intersections and map every
        # line to a set of other lines that it cuts
        for i, coord in enumerate(h_coords):
            # was (3, n)
            current = np.full((n, 3), coord)

            for j, poi in enumerate(np.cross(current, h_coords)):
                if poi[2] == 0:
                    continue

                poi[0] /= poi[2]
                poi[1] /= poi[2]

                if (min(poi[0], poi[1]) < 0 or poi[0] >= h or poi[1] >= w):
                    continue

                coords = tuple(map(int, poi[:2]))
                ipoints[i][j] = coords
                self._pois.add(coords)
                intersections[i].add(j)

        # choose all triangles by exploring whether intersection is transitive:
        # intersect(g0, g1) and intersect(g0, g1) -> intersect(g1, g2))
        # if this implication holds: calculate the barycenter
        triangles = {}
        for c0 in intersections:
            for c1, c2 in combinations(intersections[c0], 2):
                if c2 not in intersections[c1]:
                    continue

                key = tuple(sorted((c0, c1, c2)))
                if key not in triangles:

                    p0, p1, p2 = (ipoints[c0][c1],
                                  ipoints[c0][c2],
                                  ipoints[c1][c2])

                    center = np.array([np.sum(z) // 3 for z
                                       in zip(p0, p1, p2)])

                    r = 0
                    for p in (p0, p1, p2):
                        d = np.linalg.norm(center - np.array(p))
                        if d > r:
                            r = int(d * 2)

                    triangles[key] = tuple(center) + (r, )

        self._barycenter = triangles
        return self.pipeline[0].arr

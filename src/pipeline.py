# -*- coding: utf-8 -*-


from itertools import combinations
from collections import defaultdict

import numpy as np
import skimage.draw as skd
import scipy.ndimage as scnd
import skimage.transform as skt

from . import tmeasure
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

    def __sub__(self, mod):
        self._module_names.remove(mod.name)
        del self._modules[mod.name]

    def _run(self, name: str) -> None:
        mod = self._modules[name]
        if mod.disabled:
            return

        mod.arr = mod.execute()
        assert mod.arr is not None

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
        executed = tmeasure(log.info, 'finished pipeline in %sms')
        log.info('running pipeline\n' + 40 * '-')

        self._modules_executed = [self._mod_initial]
        for name in self._module_names:
            log.debug('executing module %s', name)
            ran = tmeasure(log.debug, 'execution took %sms')
            self._run(name)
            ran()

        executed()


# ---


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

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, state: bool):
        self._disabled = state

    def __init__(self, name: str):
        assert type(name) is str
        self._name = name
        self._disabled = False

    def execute(self) -> None:
        raise NotImplementedError

# ---


class Binarize(Module):

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        assert 0 <= threshold and threshold <= 255
        self._threshold = threshold

    @property
    def reference_color(self) -> tuple:
        return self._reference_color

    @reference_color.setter
    def reference_color(self, color: tuple):
        self._reference_color = color

    def __init__(self, name: str):
        super().__init__(name)
        self._threshold = 100
        self._reference_color = [180, 20, 20]

    def execute(self) -> np.ndarray:
        log.info('binarize: threshold=%f, reference=%s',
                 self.threshold, str(self.reference_color))

        src = self.pipeline[-1].arr.astype(np.int64)
        return self.apply(src)

    def apply(self, src):
        assert src.ndim == 3

        try:
            ref = self._reference_color_arr
        except AttributeError:
            ref = self._reference_color_arr = np.full(
                src.shape, self.reference_color)

        tgt = np.linalg.norm(src - ref, axis=2)
        tgt[tgt >= self.threshold] = 0
        tgt[tgt > 0] = 255
        return tgt


class Morph(Module):

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        self._iterations = int(iterations)

    def __init__(self, name: str):
        super().__init__(name)
        self._iterations = 1


class Dilate(Morph):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        log.info('dilate with %d iterations', self.iterations)

        src = self.pipeline[-1].arr
        return self.apply(src)

    def apply(self, src: np.ndarray) -> np.ndarray:
        if self.iterations == 0:
            return src

        tgt = np.zeros(src.shape)
        tgt[scnd.binary_dilation(src, iterations=self.iterations)] = 255
        return tgt


class Erode(Morph):

    def __init__(self, name: str):
        super().__init__(name)

    def execute(self) -> np.ndarray:
        log.info('erode with %d iterations', self.iterations)

        src = self.pipeline[-1].arr
        return self.apply(src)

    def apply(self, src: np.ndarray) -> np.ndarray:
        if self.iterations == 0:
            return src

        tgt = np.zeros(src.shape)
        tgt[scnd.binary_erosion(src, iterations=self.iterations)] = 255
        return tgt


# Currently unused...
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
    def src(self) -> np.ndarray:
        return self._src

    # ---

    @property
    def min_angle(self) -> int:
        return self._min_angle

    @min_angle.setter
    def min_angle(self, a: int):
        assert 0 < a and a < 90
        self._min_angle = int(a)

    @property
    def min_distance(self) -> int:
        return self._min_distance

    @min_distance.setter
    def min_distance(self, d: int):
        assert 0 < d and d < 200
        self._min_distance = int(d)

    @property
    def red_detection(self) -> int:
        return self._red_detection

    @red_detection.setter
    def red_detection(self, size: int):
        assert 0 < size
        self._red_detection = int(size)

    @property
    def patmatch_threshold(self) -> float:
        return self._patmatch_threshold

    @patmatch_threshold.setter
    def patmatch_threshold(self, t: float):
        assert 0 <= t and t <= 1
        self._patmatch_threshold = t

    # ---

    @property
    def angles(self) -> np.array:
        return self._angles

    @property
    def dists(self) -> np.array:
        return self._dists

    @property
    def pois(self) -> dict:
        return self._pois

    @property
    def barycenter(self) -> dict:
        return self._barycenter

    def _iter_intersections(self, h_coords: np.ndarray):
        h, w = self.src.shape
        n, _ = h_coords.shape

        def _check(poi):
            if poi[2] == 0:
                return False

            poi[0] /= poi[2]
            poi[1] /= poi[2]

            # adjust boundaries for truncated triangles here
            border = 1.2 * min(h, w)
            if (
                    min(poi[0], poi[1]) < -border or
                    poi[0] >= h + border or
                    poi[1] >= w + border):
                return False

            return True

        # Take each homogeneous vector and calculate possible
        # intersections with all other vectors
        for i, coord in enumerate(h_coords):
            current = np.full((n, 3), coord)

            for j, poi in enumerate(np.cross(current, h_coords)):
                if _check(poi):
                    yield i, j, tuple(map(int, poi[:2]))

    def _calc_intersections(self, h_coords: np.ndarray):
        """
        Calculate points of intersections and map every
        line to a set of other lines that it cuts.

        This method returns two dictionaries (intersections, ipoints)

          - intersections maps line index to line index
              (int -> int)
          - ipoints maps each cut to their point of intersection
              (int -> int -> tuple(3))

        """
        binarized = self.pipeline['pre_dilate'].arr
        off = self.red_detection // 2

        for i, j, coords in self._iter_intersections(h_coords):
            y, x = coords

            # abort if there's no red to be found on
            # the point of intersection
            if not np.any(binarized[y-off:y+off, x-off:x+off]):
                fmt = 'discarding %s because it lacks red'
                log.debug(fmt, str(coords))
                continue

            a, b = min(i, j), max(i, j)
            self._pois[a][b] = coords

    def _iter_triangles(self):
        triangles = set()
        points = self.pois

        for c0 in points:
            for c1, c2 in combinations(points[c0].keys(), 2):

                # it is very imported to use them sorted
                # as the point datastructure uses the same order
                key = tuple(sorted((c0, c1, c2)))

                # the order of theses tests are important as the lookup for
                # d[i] in a defaultdict creates the default object
                if key not in triangles and (
                        c1 in points.keys() and
                        c1 in points[c0].keys() and
                        c2 in points[c0].keys() and
                        c2 in points[c1].keys()):

                    p0, p1, p2 = (points[c0][c1],
                                  points[c0][c2],
                                  points[c1][c2])

                    triangles.add(key)
                    yield key, (p0, p1, p2)

    def _calc_triangles(self):
        """
        Choose all triangles by exploring whether intersection is transitive:
          - intersect(g0, g1) and intersect(g0, g1) -> intersect(g1, g2))

        Only consider those, where the area around the point of intersection
        contains some red color values.

        If these implications hold: calculate the barycenter.

        The resulting triangles are again filtered to only contain those
        which are likely to resemble a yield sign.

        """
        triangles = {}

        for key, points in self._iter_triangles():

            # calculate the (bary)center by taking the
            # sum of all vertices and divide them by
            # their amount (always 3 in this case)
            center = np.array([np.sum(z) // 3 for z in zip(*points)])
            triangles[key] = tuple(center)

        # finally, filter found triangles
        log.info('found %d triangles', len(triangles))
        done = tmeasure(log.debug,
                        'filtering took %sms and %d triangles remain')

        self._barycenter = self._filter_triangles(triangles)
        done(len(self.barycenter))

    def _filter_triangles(self, triangles):
        """
        Filter triangles by their appearance. A yield sign is an upside
        down triangle and we assume they are not too skewed by
        perspective as road signs are usually turned towards the
        driver.

        """
        pois = self.pois
        filtered = {}

        def timing(gen):
            for it in gen:
                done = tmeasure(log.debug, ' it took %sms')
                yield it
                done()

        for key, barycenter in timing(triangles.items()):

            l1, l2, l3 = key
            points = pois[l1][l2], pois[l1][l3], pois[l2][l3]
            ycoords, xcoords = zip(*points)

            log.debug('looking at %s with %s', key, points)

            # take the quadratic area spanned by the triangle
            y0, y1 = np.min(ycoords), np.max(ycoords)
            x0, x1 = np.min(xcoords), np.max(xcoords)

            h, w = y1 - y0, x1 - x0
            img_h, img_w = self._src.shape

            # abort if the sign size does not fit
            if img_h/2 < h or h < 10 or img_w/2 < w or w < 10:
                log.debug('discarding %s %dx%d sized triangle', key, h, w)
                continue

            # abort if the proportions do not fit
            ratio = min(h, w) / max(h, w)
            if ratio < .5:
                log.debug('discarding %s with ratio %f', key, ratio)
                continue

            # apply some pattern matching
            frame = y0, y1, x0, x1
            if not self._patmatch((h, w), frame, (ycoords, xcoords)):
                log.debug('discarding %s because patterns do not match', key)
                continue

            filtered[key] = barycenter

        return filtered

    def _patmatch(self, dim, frame, coords):
        ref = np.zeros(dim, dtype=np.bool)
        tri = np.zeros(dim, dtype=np.bool)

        h, w = dim
        y0, y1, x0, x1 = frame
        ycoords, xcoords = coords

        done = tmeasure(log.debug, 'drawing polygons took %sms')

        # the reference triangle is an equilateral triangle
        # pointing downwards
        rr, cc = skd.polygon([-1, -1, h], [-1, w, w/2])
        ref[rr, cc] = 1

        # now apply the found triangle
        rr, cc = skd.polygon(
            ycoords - np.min(ycoords),
            xcoords - np.min(xcoords))
        tri[rr, cc] = 1

        done()

        # everybody walk the pattern match
        d = np.sum(ref ^ tri) / ref.size
        log.debug('pattern matching delta: %f', d)
        return d < self.patmatch_threshold

    def __init__(self, name: str):
        super().__init__(name)

        # defaults
        self.min_angle = 45
        self.min_distance = 100
        self.red_detection = 15
        self.patmatch_threshold = 0.2

    def execute(self) -> np.ndarray:
        self._src = self.pipeline[-1].arr
        tgt = self.pipeline[0].arr

        h, w = self._src.shape

        # reset
        self._pois = defaultdict(dict)  # points of intersections
        self._barycenter = {}           # triangles, mapped by tuple -> tuple
        self._angles = []
        self._dists = []

        # --- apply hough transformation

        done = tmeasure(log.debug, ' skt hough: %sms, found %d lines')

        try:
            _, angles, dists = skt.hough_line_peaks(
                *skt.hough_line(self.src),
                min_angle=self.min_angle,
                min_distance=self.min_distance)

        # this is thrown if no peaks are found
        except IndexError:
            angles = []

        done(len(angles))

        if len(angles) == 0:
            return tgt

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

        # populate self.pois
        self._calc_intersections(h_coords)

        # populate self.barycenter by calculating
        # and filtering triangles
        self._calc_triangles()

        return tgt

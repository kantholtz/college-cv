#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


ref_points = np.array([[0, 0, 1],
                       [0, 3, 1],
                       [2, 0, 1]])
print('ref_points')
print(ref_points)

vectors = np.array([[1, 1],
                    [1, 0],
                    [0, 1]])
print('vectors')
print(vectors)

line_points = np.copy(ref_points)
line_points[:, :2] += vectors
print('line_points')
print(line_points)

h_coords = np.cross(ref_points, line_points)
print('h_coords')
print(h_coords)

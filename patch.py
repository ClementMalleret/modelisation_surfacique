import numpy as np
import random
from itertools import product


# Utilitary functions for the Patch class
def binom(n, k):
    assert(0 <= k <= n)
    if k == 0:
        return 1
    else:
        return (n-(k-1)) * binom(n, k-1) // k

def bernstein(t, n, j):
    return binom(n, j)*t**j * (1-t)**(n-j)

def casteljau(p, t):
    n = len(p)
    return sum([bernstein(t, n-1, j) * p[j] for j in range(n)])


class Patch:
    def __init__(self):
        self.control_points = np.zeros(shape=(4, 4, 3))

    def __str__(self):
        out = ""
        for i in range(4):
            for j in range(4):
                x, y, z = self.control_points[i, j]
                out += "   %s   %s   %s\n" % (x, y, z)
        return out

    def __getitem__(self, key):
        i, j = key
        return self.control_points[i, j]
    
    def __setitem__(self, key, val):
        i, j = key
        self.control_points[i, j] = val

    def get_surface(self, Nx=30, Ny=30):
        points = np.zeros(shape=(Nx * Ny, 3))
        for i, j in product(range(Nx), range(Ny)):
            s = i * 1/Nx
            t = j * 1/Ny
            points[i + j * Nx] = self.evaluate(s, t)
        return points

    def randomize(self, min_x, max_x, min_y, max_y, min_z=0, max_z=1):
        x_step = (max_x - min_x) / 3
        y_step = (max_y - min_y) / 3
        for i, j in product(range(4), repeat=2):
            x = min_x + i * x_step
            y = min_y + j * y_step
            z = random.uniform(min_z, max_z)
            self.control_points[i, j] = np.array([x, y, z])

    def evaluate(self, s, t):
        line = np.zeros(shape=(4, 3))
        for column_index in range(4):
            column = self.control_points[column_index]
            line[column_index] = casteljau(column, t)

        return casteljau(line, s)

    # def get_normal_field(self, x_parameter, y_parameter):
    #     normal_field = np.zeros(shape=(len(x_parameter), len(y_parameter), 3))

    #     for x, y in product(x_parameter, y_parameter):
    #         normal_field[x, y] = self.evaluate

    #     return normal_field

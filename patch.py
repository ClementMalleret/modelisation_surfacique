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

# Base change matrix, used to go from the Bernstein base to the polynomial one.
M = np.matrix('1 0 0 0; -3 3 0 0; 3 -6 3 0; -1 3 -3 1')


class Patch:
    """
    Class representing a Bézier surface.
    A patch is a Bézier surface containing 4x4 control points.
    """
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
        """
        Returns an array of points of the surface, evaluated regularily with
        Nx points on the x axis, and Ny points on the y axis.
        """
        points = np.zeros(shape=(Nx * Ny, 3))
        for i, j in product(range(Nx), range(Ny)):
            s = i * 1/Nx
            t = j * 1/Ny
            points[i + j * Nx] = self.evaluate(s, t)
        return points

    def randomize(self, min_x, max_x, min_y, max_y, min_z=0, max_z=1):
        """
        Randomize the values of the patch.
        Only the height is truly randomized, the x and y coordinates are generated regularily
        between their given min and max.
        """
        x_step = (max_x - min_x) / 3
        y_step = (max_y - min_y) / 3
        for i, j in product(range(4), repeat=2):
            x = min_x + i * x_step
            y = min_y + j * y_step
            z = random.uniform(min_z, max_z)
            self.control_points[i, j] = np.array([x, y, z])

    def evaluate(self, s, t):
        """
        Evaluate the surface at the given parameters.
        Uses the 1D DeCasteljau algorithm applied to the columns, then to the resulting line.
        """
        line = np.zeros(shape=(4, 3))
        for column_index in range(4):
            column = self.control_points[column_index]
            line[column_index] = casteljau(column, t)

        return casteljau(line, s)

    def evaluate_partial_derivative(self, parameter_index, u, v):
        """
        Computes the partial derivative.
        The variable used for the derivative is chose from the 'parameter_index'.
        Parameter_index = 1: we use the first variable, u.
        Parameter_index = 2: we use the second variable, v.

        Returns a 3D point.
        """
        point = np.zeros(shape=(3))

        # choice of the variable we use for the derivative: index=1 -> u, index=2 -> v
        if parameter_index == 0:
            U = np.array([[0, 1, 2*u, 3*u**2]]) # derivative of [1, x, x**2, x**3]
            VT = np.transpose(np.array([[v**i for i in range(4)]]))
        else:
            U = np.array([[u**i for i in range(4)]])
            VT = np.transpose(np.array([[0, 1, 2*v, 3*v**2]]))

        for k in range(3):
            point[k] = U * M * self.control_points[:,:,k] * np.transpose(M) * VT

        return point

    def get_normal_field(self, x_parameter, y_parameter):
        """
        Returns the normal field of the surface, on the product of the given coordinate arrays,
        x_parameter and y_parameter.
        """
        normal_field = np.zeros(shape=(len(x_parameter), len(y_parameter), 3))

        for i, x in enumerate(x_parameter):
            for j, y in enumerate(y_parameter):
                Xu = self.evaluate_partial_derivative(0, x, y)
                Xv = self.evaluate_partial_derivative(1, x, y)
                cross_product = np.cross(Xu, Xv)
                normal_field[i, j] = cross_product / np.linalg.norm(cross_product)

        return normal_field
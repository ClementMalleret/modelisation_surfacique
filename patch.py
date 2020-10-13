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
        for j in range(4):
            for i in range(4):
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
            s = i * 1/(Nx - 1)
            t = j * 1/(Ny - 1)
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

    def evaluate_first_order_partial_derivative(self, parameter_index, u, v):
        """
        Computes the first order partial derivative.
        The variable used for the derivative is chose from the 'parameter_index'.
        Parameter_index = 0: we use the first variable, u.
        Parameter_index = 1: we use the second variable, v.

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

    def evaluate_second_order_derivative(self, parameter_index_1, parameter_index_2, u, v):
        """
        Computes the second order derivative.
        The variable used for the derivative is chose from the 'parameter_index_i'.
        Parameter_index_i = 0: we use the first variable, u.
        Parameter_index_i = 1: we use the second variable, v.

        Returns a 3D point.
        """
        point = np.zeros(shape=(3))

        # choice of the variable we use for the derivative: index=1 -> u, index=2 -> v
        if parameter_index_1 == 0 and parameter_index_2==0:
            U = np.array([[0, 0, 2, 6*u]]) 
            VT = np.transpose(np.array([[v**i for i in range(4)]]))

        elif parameter_index_1 == 1 and parameter_index_2==1:
            U = np.array([[u**i for i in range(4)]])
            VT = np.transpose(np.array([[0, 0, 2, 6*v]]))

        else:
            U = np.array([[0, 1, 2*u, 3*u**2]])
            VT = np.transpose(np.array([[0, 1, 2*v, 3*v**2]]))

        for k in range(3):
            point[k] = U * M * self.control_points[:,:,k] * np.transpose(M) * VT

        return point

    def evaluate_normal(self, u, v):
        """
        Evaluates the normal vector to the surface, at the given parameters.
        """
        Xu = self.evaluate_first_order_partial_derivative(0, u, v)
        Xv = self.evaluate_first_order_partial_derivative(1, u, v)
        cross_product = np.cross(Xu, Xv)
        return cross_product / np.linalg.norm(cross_product)

    def get_normal_field(self, x_parameter, y_parameter):
        """
        Returns the normal field of the surface, on the product of the given coordinate arrays,
        x_parameter and y_parameter.
        """
        normal_field = np.zeros(shape=(len(x_parameter), len(y_parameter), 3))

        for i, x in enumerate(x_parameter):
            for j, y in enumerate(y_parameter):
                normal_field[i, j] = self.evaluate_normal(x, y)

        return normal_field

    def compute_isophote(self, L, c, epsilon=0.02, x_param=None, y_param=None):
        """
        Computes the isophote line for the given direction L, where L is an 3D vector, and
        for the given brightness c.
        """
        if x_param is None:
            x_param = np.arange(0, 1.01, 0.01)
        if y_param is None:
            y_param = np.arange(0, 1.01, 0.01)

        normal_field = self.get_normal_field(x_param, y_param)
        isophote = []

        for i, j in product(range(len(x_param)), range(len(y_param))):
            if abs(np.dot(normal_field[i, j], L) - c) < epsilon:
                isophote.append(self.evaluate(x_param[i], y_param[j]))
        return isophote

    def get_first_fundamental_form(self, u, w):
        """
        Returns the first fundamental form at the given parameters.
        """
        mat = np.zeros(shape=(2,2))
        for i in range(2):
            for j in range(2):
                first_var = self.evaluate_first_order_partial_derivative(i, u, w)
                second_var = self.evaluate_first_order_partial_derivative(j, u, w)
                mat[i, j] = np.dot(first_var, second_var)
        return mat

    def get_second_fundamental_form(self, u, w):
        """
        Returns the second fundamental form at the given parameters.
        """
        mat = np.zeros(shape=(2,2))
        N = self.evaluate_normal(u, w)

        for i in range(2):
            for j in range(2):
                derivative = self.evaluate_second_order_derivative(i, j, u, w)
                mat[i, j] = np.dot(derivative, N)
        return mat

    def evaluate_principal_curvature(self, u, w):
        """
        Returns an np.array containing the principal curvatures at the given parameters.
        """
        G = self.get_first_fundamental_form(u, w)
        H = self.get_second_fundamental_form(u, w)
        L = H * np.linalg.inv(G)
        # eig returns 2 arrays: the first contains the eigenvalues (that we want).
        # The second contains the eigenvectors. That's why we only take the first element.
        return np.linalg.eig(L)[0]

    def evaluate_gauss_curvature(self, Nx, Ny):
        points = np.zeros(shape=(Nx * Ny))
        for i, j in product(range(Nx), range(Ny)):
            s = i * 1/(Nx - 1)
            t = j * 1/(Ny - 1)
            point = np.prod(self.evaluate_principal_curvature(s, t))
            points[i + j * Nx] =  point
        return points

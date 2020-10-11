import numpy as np
from math import floor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy

BezierControlPoints = np.loadtxt("surface3")

Nx, Ny = 30, 30
Lx, Ly = 2, 2
# print(BezierSurf)


def display(curve):
    xline = curve[:, 0]
    yline = curve[:, 1]
    zline = curve[:, 2]

    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot_trisurf(xline, yline, zline, cmap='viridis', edgecolor='none')
    plt.show()


# display(BezierControlPoints)


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


def fill_good_array(points):  # create a nice representation of the list-like read file
    good_array = np.zeros(shape=(4, 4, 3))
    for i in range(4):
        for j in range(4):
            z = points[4*i+j, 2]
            y = points[4*i+j, 1]
            x = points[4*i+j, 0]
            good_array[i, j] = np.array([x, y, z])
    return good_array


def get_heigth_map(good_array):
    return good_array[:, :, 2]


def create_surface(control_points):  # optimized
    bezier_input = get_heigth_map(control_points)

    line_tranformed = np.zeros(shape=(len(bezier_input), Ny))
    for i, line in enumerate(bezier_input):
        print(line)
        line_tranformed[i] = np.array([casteljau(line, t)
                                       for t in np.linspace(0, 1, Nx)])
    column_tranformed = np.transpose(line_tranformed)
    surface_bezier = np.zeros(shape=(Nx, Ny))
    for i, column in enumerate(column_tranformed):
        surface_bezier[i] = np.array([casteljau(column, s)
                                      for s in np.linspace(0, 1, Ny)])
    return surface_bezier


def create_meshgrid(heigth_map, Lx, Ly, Nx, Ny):
    X = np.linspace(0, Lx, Nx)
    Y = np.linspace(0, Ly, Ny)

    points_list = np.zeros(shape=(Nx * Ny, 3))
    for i in range(Nx):
        for j in range(Ny):
            x = X[i]
            y = Y[j]
            z = heigth_map[i][j]
            points_list[i*Ny + j] = np.array([x, y, z])
    return points_list

# good_array = fill_good_array(BezierControlPoints)
# display(create_meshgrid(create_surface(good_array), Lx, Ly, Nx, Ny))

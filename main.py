#!/usr/bin/env python3

import numpy as np

from surface import Surface

def surfaces_generation():
    """
    Code used to generate the surfaces.
    """
    surf = Surface()
    surf.randomize(1, 1)
    surf.save_in_file('surface2')

    surf = Surface()
    surf.randomize(2, 2)
    surf.save_in_file('surface3')

    surf = Surface()
    surf.randomize(1, 3)
    surf.save_in_file('surface4')

def plot_surface(surface_name):
    """
    Code used to plot a given surface
    """
    surf = Surface()
    surf.load(surface_name)
    surf.plot(30, 30)

def plot_isophotes(surface_name, L, epsilon):
    """
    Code used to plot the isophotes to a given surface
    """
    surf = Surface()
    surf.load(surface_name)
    surf.plot_isophotes(L, np.arange(0, 1, 0.1), epsilon=epsilon)

def plot_curvature(surface_name):
    """
    Code used to plot the curvature for a given surface
    """
    surf = Surface()
    surf.load(surface_name)
    surf.plot_curvature(30, 30)

# surfaces_generation()

# plot_surface("surface1")
# plot_surface("surface2")
# plot_surface("surface3")
# plot_surface("surface4")

# plot_isophotes("surface1", np.array([1, 0, 0]), 0.01)
# plot_isophotes("surface1", np.array([0, 1, 0]), 0.01)
# plot_isophotes("surface1", np.array([1, 1, 0]), 0.01)
# plot_isophotes("surface1", np.array([1, 1, 1]), 0.01)

# plot_isophotes("surface2", np.array([1, 0, 0]), 0.002)
# plot_isophotes("surface2", np.array([0, 1, 0]), 0.002)
# plot_isophotes("surface2", np.array([1, 1, 0]), 0.002)
# plot_isophotes("surface2", np.array([1, 1, 1]), 0.002)

# plot_isophotes("surface3", np.array([1, 0, 0]), 0.002)
# plot_isophotes("surface3", np.array([0, 1, 0]), 0.002)
# plot_isophotes("surface3", np.array([1, 1, 0]), 0.002)
# plot_isophotes("surface3", np.array([1, 1, 1]), 0.002)

# plot_isophotes("surface4", np.array([1, 0, 0]), 0.002)
# plot_isophotes("surface4", np.array([0, 1, 0]), 0.002)
# plot_isophotes("surface4", np.array([1, 1, 0]), 0.002)
# plot_isophotes("surface4", np.array([1, 1, 1]), 0.002)

# plot_curvature("surface1")
# plot_curvature("surface2")
# plot_curvature("surface3")
# plot_curvature("surface4")

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as col
from mpl_toolkits.mplot3d import Axes3D

from patch import Patch


class Surface:
    """
    Class representing a Bézier surface as a set of given patches.
    """
    def __init__(self, length=0, width=0):
        self.patches = []

    def __iter__(self):
        for patch in self.patches:
            yield patch
    
    def add_patch(self, patch):
        self.patches.append(patch)

    def randomize(self, patch_length, patch_width, min_height=0, max_height=0.2):
        """
        Creates a random Bézier surface. The resulting surface will be C0 only.
        patch_length is the number of patches along the x axis that will be generated.
        patch_width is the number of patches along the y axis that will be generated.
        """
        self.patches = []
        for w, l in product(range(patch_width), range(patch_length)):
            # Patches are created and stored in a 1D array.
            # The patch at coordinate (x, y) can be retrieved as self.patches[x + y * patch_length]
            # However this structure is true ONLY in this generation fuction, as in general, if
            # we import another surface, it may not follow this particular pattern.
            # It is however useful in this generation, as it allows us to easily correct random
            # patches to make them c0, without checking each control points 1 by 1.
            patch = Patch()
            patch.randomize(min_x = l, max_x = l + 1, 
                            min_y = w, max_y = w + 1, 
                            min_z = min_height, max_z = max_height)
            self.patches.append(patch)
        
        # make the connections continuous
        for l, w in product(range(1, patch_length), range(patch_width)):
            previous_patch = self.patches[(l - 1) + w * patch_length]
            current_patch = self.patches[l + w * patch_length]
            for i in range(4):
                current_patch[0, i] = previous_patch[3, i]
        
        for l, w in product(range(patch_length), range(1, patch_width)):
            previous_patch = self.patches[l + (w - 1) * patch_length]
            current_patch = self.patches[l + w * patch_length]
            for i in range(4):
                current_patch[i, 0] = previous_patch[i, 3]

    def load(self, filename):
        """
        Loads a surface from a text file.
        The file must have the following structure:
        Each line is a control point. It contains all 3 coordinates, separated by spaces only.
        The file must contain n * 16 control points, where n is a positive integer.
        """
        self.patches=[]

        control_points = np.loadtxt(filename)
        nb_of_patches = int(len(control_points) / 16)

        patches_control_points = np.split(control_points, nb_of_patches)

        for patch_nb in range(len(patches_control_points)):
            patch = Patch()
            patch_control_points = patches_control_points[patch_nb]

            for i in range(4):
                for j in range(4):
                    z = patch_control_points[4*i+j, 2]
                    y = patch_control_points[4*i+j, 1]
                    x = patch_control_points[4*i+j, 0]
                    patch.control_points[i, j] = np.array([x, y, z])
            self.patches.append(patch)

    def save_in_file(self, filename):
        """
        Save the surface in a text file.
        To see the file format, please refer to the 'load' method.
        """
        with open(filename, 'w') as f:
            for patch in self:
                f.write(str(patch))
                f.write("\n")

    def draw_to(self, ax, Nx_per_patch = 20, Ny_per_patch = 20, color='gray', **kwargs):
        """
        Draws the current surface to the given axes.

        Any additional keyword argument, contained in kwargs, will be passed to the plotting
        function. You can for example pass a custom colormap, facecolors, transparency, etc.

        Please refer to the documentation of matplotlib's "plot_surface" function for a complete
        list of possible keyword arguments.
        """
        for patch in self:
            patch.draw_to(ax, Nx_per_patch, Ny_per_patch, color=color, **kwargs)

    def plot(self, Nx_per_patch = 20, Ny_per_patch = 20, cmap=None):
        """
        Plots the surface, with the given number of points per axis and per patch.
        """
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        self.draw_to(ax, Nx_per_patch, Ny_per_patch, cmap=cmap)
        plt.show()

    def draw_isophote_to(self, ax, color, L, c, epsilon, x_param, y_param):
        """
        Computes the isophote line for the given direction L, where L is an 3D vector, and
        for the given brightness c.
        """
        for patch in self:
            patch.draw_isophote_to(ax, color, L, c, epsilon, x_param, y_param)

    def plot_isophotes(self, L, c, epsilon=0.01, x_param=None, y_param=None, cmap='viridis'):
        """
        Computes and plots the isophote line for the given direction L, where L is an 3D vector, and
        for the given brightnesses c (c is a vector of brignesses).

        x_param and y_param are custom coordinates array that can be passed. If passed, the isophotes
        will be calculated on the cartesian product of those arrays, instead of on [0, 0.01, ..., 1]².
        
        cmap is the color map used to display the surface. If None is passed, the surface will be
        displayed in gray.
        """
        if x_param is None:
            x_param = np.arange(0, 1.01, 0.01)
        if y_param is None:
            y_param = np.arange(0, 1.01, 0.01)

        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        colors = cm.tab20(c)

        for brightness, color in zip(c, colors):
            self.draw_isophote_to(ax, color, L, brightness, epsilon, x_param, y_param)
        self.draw_to(ax, cmap=cmap, alpha=0.5)

        plt.show()

    def plot_curvature(self, Nx_per_patch, Ny_per_patch):
        """
        Plots the curvature of the surface.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Get all the curvatures
        curvatures = []
        for patch in self:
            curvatures.append(patch.evaluate_abs_curvature(Nx_per_patch, Ny_per_patch))

        # Map the curvatures to colors
        norm=col.Normalize(vmin=np.amin(curvatures), vmax=np.amax(curvatures))
        cmap = cm.Spectral_r(norm(curvatures))

        # Plot every patch with their colors
        for patch, colors in zip(self, cmap):
            patch.draw_to(ax, 30, 30, facecolors=colors)
        
        # add colorbar
        m = cm.ScalarMappable(cmap=cm.Spectral_r, norm=norm)
        m.set_array(curvatures)
        fig.colorbar(m)

        plt.show()

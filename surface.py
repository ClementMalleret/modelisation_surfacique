import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
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

    def randomize(self, patch_length, patch_width, min_height=0, max_height=1):
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


    def draw_to(self, ax, Nx_per_patch = 20, Ny_per_patch = 20, alpha=1):
        total_surface = None
        for patch in self:
            patch_surface = patch.get_surface(Nx_per_patch, Ny_per_patch)
            if total_surface is None:
                total_surface = patch_surface
            else:
                total_surface = np.concatenate((total_surface, patch_surface))
        
        xline = total_surface[:, 0]
        yline = total_surface[:, 1]
        zline = total_surface[:, 2]

        ax.plot_trisurf(xline, yline, zline, cmap='viridis', edgecolor='none', alpha=alpha)

    def plot(self, Nx_per_patch = 20, Ny_per_patch = 20):
        """
        Plots the surface, with the given number of points per axis and per patch.
        """
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        self.draw_to(ax, Nx_per_patch, Ny_per_patch)
        plt.show()

    def compute_isophote(self, L, c):
        """
        Computes the isophote line for the given direction L, where L is an 3D vector, and
        for the given brightness c.
        """
        isophote = []
        for patch in self:
            isophote += patch.compute_isophote(L, c)
        return isophote

    def plot_isophote(self, L, c):
        """
        Computes and plots the isophote line for the given direction L, where L is an 3D vector, and
        for the given brightness c.
        """
        isophotes = self.compute_isophote(L, c)
        
        if not isophotes:
            print("No isophote found for the given parameters.")
            return

        xline = [point[0] for point in isophotes]
        yline = [point[1] for point in isophotes]
        zline = [point[2] for point in isophotes]

        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        self.draw_to(ax, alpha=0.6)
        ax.scatter3D(xline, yline, zline, c='red')
        
        plt.show()

    def plot_curvature(self, Nx_per_patch, Ny_per_patch):
        """
        Plots the curvature of the surface, using the gauss curvature.
        """
        curvature_map = None
        for patch in self:
            patch_curvature_map = patch.evaluate_gauss_curvature(Nx_per_patch, Ny_per_patch)
            if curvature_map is None:
                curvature_map = patch_curvature_map
            else:
                curvature_map = np.concatenate((curvature_map, patch_curvature_map))

        total_surface = None
        for patch in self:
            patch_surface = patch.get_surface(Nx_per_patch, Ny_per_patch)
            if total_surface is None:
                total_surface = patch_surface
            else:
                total_surface = np.concatenate((total_surface, patch_surface))

        # Unfortunately, plot_trisurf doesn't support the facecolor keyword argument, that we
        # need to plot the color map.
        # (PR still in draft since 2018, yay https://github.com/matplotlib/matplotlib/pull/12073 )
        # So we need to convert our surface to a format supported by plot_surface, which need a
        # 2D array of the values. So we have to reorganize all our surface to make it work.
        # It is long and ugly, but it works.
        # A much simplier solution would have been to show the curvature as points using scatter3D,
        # which uses the same format of points as plot_trisurf and supports facecolors.
        # However I spend too much time and effort in this solution to abandon it.

        # plot_surface works by using an array of array. The inner arrays corresponds to the lines:
        # Thus, to acces an element, we do array[y][x], or array[y, x] if we use a numpy array.
        # So do not be confused if we do mat[y, x] = val(x, y) in the following.

        # Computing the necessary parameters
        nb_points_per_patch = Nx_per_patch * Ny_per_patch

        x_min_patch = min(total_surface[:nb_points_per_patch, 0])
        x_max_patch = max(total_surface[:nb_points_per_patch, 0])
        y_min_patch = min(total_surface[:nb_points_per_patch, 1])
        y_max_patch = max(total_surface[:nb_points_per_patch, 1])

        x_size_patch = x_max_patch - x_min_patch
        y_size_patch = y_max_patch - y_min_patch

        x_min = min(total_surface[:, 0])
        x_max = max(total_surface[:, 0])
        y_min = min(total_surface[:, 1])
        y_max = max(total_surface[:, 1])           

        x_step = x_size_patch / (Nx_per_patch - 1)
        y_step = y_size_patch / (Ny_per_patch - 1)

        # Initializing the resulting arrays
        X = np.arange(x_min, x_max + x_step / 2, x_step)
        Y = np.arange(y_min, y_max + y_step / 2, y_step)
        Z = np.zeros(shape=(len(Y), len(X)))        
        X, Y = np.meshgrid(X, Y)

        curvature = np.zeros(shape=(len(Y), len(X)))

        # Reorganizing data
        for point, curvature_val in zip(total_surface, curvature_map):
            x, y, z = point
            i = int(round((x - x_min) / x_step))
            j = int(round((y - y_min) / y_step))

            Z[j, i] = z
            curvature[j, i] = curvature_val

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # add surface
        cmap = cm.coolwarm(curvature)
        ax.plot_surface(X, Y, Z, facecolors=cmap)
        
        # add colorbar
        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(curvature)
        fig.colorbar(m)

        plt.show()

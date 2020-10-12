import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from patch import Patch


class Surface:
    def __init__(self, length=0, width=0):
        self.patches = []

    def __iter__(self):
        for patch in self.patches:
            yield patch
    
    def add_patch(self, patch):
        self.patches.append(patch)

    def randomize(self, patch_length, patch_width, min_height=0, max_height=1):
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
        with open(filename, 'w') as f:
            for patch in self:
                f.write(str(patch))
                f.write("\n")

    def plot(self):
        total_surface = None
        for patch in self:
            patch_surface = patch.get_surface()
            if total_surface is None:
                total_surface = patch_surface
            else:
                total_surface = np.concatenate((total_surface, patch_surface))
        
        xline = total_surface[:, 0]
        yline = total_surface[:, 1]
        zline = total_surface[:, 2]

        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.plot_trisurf(xline, yline, zline, cmap='viridis', edgecolor='none')
        plt.show()

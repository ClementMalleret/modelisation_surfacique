import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt

from shareable import casteljau


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


# surf = Surface()
# surf.randomize(1, 1)
# surf.save_in_file('surface2')

# surf = Surface()
# surf.randomize(2, 2)
# surf.save_in_file('surface3')

# surf = Surface()
# surf.randomize(1, 3)
# surf.save_in_file('surface4')

surf = Surface()
surf.load("surface1")
# surf.randomize(2, 2)
surf.plot()

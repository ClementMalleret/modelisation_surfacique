import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt

from shareable import casteljau


class Patch:
    def __init__(self, x, y, min_height=0, max_height=1):
        self.control_points = np.zeros(shape=(4, 4, 3))
        for i in range(4):
            for j in range(4):
                self.control_points[i, j] = np.array([x + i * 1/3, y + j * 1/3, 0])

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

    def randomize(self, min_height=0, max_height=1):
        for i, j in product(range(4), repeat=2):
            self.control_points[i, j, 2] = random.uniform(min_height, max_height)

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
        self.length = length
        self.width = width
        self.patches = [[Patch(x, y) for y in range(width)] for x in range(length)]

    def __iter__(self):
        for l, w in product(range(self.length), range(self.width)):
            yield self.patches[l][w]
    
    def add_patch(self, x, y, patch):
        self.patches[x, y] = patch

    def randomize(self, min_height=0, max_height=1):
        for patch in self:
            patch.randomize(min_height, max_height)

    def load(self, filename):
        self.patches=[[]]

        control_points = np.loadtxt(filename)
        nb_of_patches = int(len(control_points) / 16)
        self.length = 1
        self.width = nb_of_patches

        patches_control_points = np.split(control_points, nb_of_patches)

        for patch_nb in range(len(patches_control_points)):
            print("plop")
            patch = Patch(patch_nb, 0)
            patch_control_points = patches_control_points[patch_nb]
            print(patch_control_points)

            for i in range(4):
                for j in range(4):
                    z = patch_control_points[4*i+j, 2]
                    y = patch_control_points[4*i+j, 1]
                    x = patch_control_points[4*i+j, 0]
                    patch.control_points[i, j] = np.array([x, y, z])
            self.patches[0].append(patch)
            print(len(self.patches[0]))

    def make_c0(self):
        # make the connections continuous
        for l, w in product(range(1, self.length), range(self.width)):
            previous_patch = self.patches[l - 1][w]
            current_patch = self.patches[l][w]
            for i in range(4):
                current_patch[0, i] = previous_patch[3, i]
        
        for l, w in product(range(self.length), range(1, self.width)):
            previous_patch = self.patches[l][w - 1]
            current_patch = self.patches[l][w]
            for i in range(4):
                current_patch[i, 0] = previous_patch[i, 3]

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


# surf = Surface(1, 1)
# surf.randomize()
# surf.make_c0()
# surf.save_in_file('surface2')

# surf = Surface(2, 2)
# surf.randomize()
# surf.make_c0()
# surf.save_in_file('surface3')

# surf = Surface(1, 3)
# surf.randomize()
# surf.make_c0()
# surf.save_in_file('surface4')

# surf = Surface(1, 3)
# surf.randomize()
# surf.make_c0()
# surf.save_in_file('test')

surf = Surface()
surf.load("surface3")
# surf.randomize()
# surf.make_c0()
surf.plot()
# print(surf.patches[0][0])
# print("================")
# print(surf.patches[0][0].evaluate(1, 0))

#TODO: pb de dim dans Surface: passer a un stockage 1D et foutre make_c0 dans randomize() pour pouvoir avoir tjrs acces aux dims 2D
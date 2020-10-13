#!/usr/bin/env python3

import numpy as np

from surface import Surface


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
surf.load("surface4")
# surf.randomize(1, 2)
# for patch in surf:
#     print(patch.get_normal_field([0, 0.5, 1], [0, 0.5, 1]))
# surf.plot()
# print(surf.patches[0].get_normal_field(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)))
# surf.plot_isophote(np.array([1, 1, 0]), 0.5)
# print(surf.patches[0].evaluate_principal_curvature(0.5, 0.5))
surf.plot_curvature(30, 30)

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
surf.load("teapot")
# surf.randomize(1, 2)

# surf.plot(10, 10)
# surf.plot_isophote(np.array([1, 1, 0]), range(10), epsilon=0.02)
surf.plot_curvature(30, 30)

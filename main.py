#!/usr/bin/env python3

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
# surf.load("surface1")
surf.randomize(1, 1)
for patch in surf:
    print(patch.get_normal_field([0, 0.5, 1], [0, 0.5, 1]))
print('ok')

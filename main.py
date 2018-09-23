import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pattern import pattern_gen
import scalify
from mask import sml_mask, med_mask

# input = np.array([[0,0,0,0,0,0,0,0,0],
#                   [0,1,1,0,0,1,1,0,0],
#                   [0,1,0,1,1,1,0,1,1],
#                   [1,1,1,0,1,1,1,0,1],
#                   [0,0,0,0,0,0,0,0,0]])

dims = (25, 25)

input = pattern_gen(dims)
output = scalify.scalify(input, med_mask)

fig, ax = plt.subplots()
im = ax.pcolormesh(output, cmap='YlOrBr')

plt.axis('equal')
plt.show()

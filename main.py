import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scalify
from mask import sml_mask, med_mask

input = [[0,0,0,0,0],
         [0,1,1,0,0],
         [0,1,0,1,1],
         [1,1,1,0,1],
         [0,0,0,0,0]]

dimensions = (40, 25)

output = scalify.scalify(input, med_mask)

fig, ax = plt.subplots()
im = ax.pcolormesh(output, cmap='YlOrBr')

plt.axis('equal')
plt.show()

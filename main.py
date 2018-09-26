import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit

from pattern import pattern_gen
import scalify
from mask import sml_mask, med_mask

# input = np.array([[0,0,0,0,0,0,0,0,0],
#                   [0,1,1,0,0,1,1,0,0],
#                   [0,1,0,1,1,1,0,1,1],
#                   [1,1,1,0,1,1,1,0,1],
#                   [0,0,0,0,0,0,0,0,0]])

dims = (80,50)
#dims = (6,8)

input = pattern_gen(dims, iter_n=20)

output = scalify.scalify(input, med_mask)

fig, ax = plt.subplots()
im = ax.pcolormesh(output, cmap='YlOrBr')
ax.axis('off')

plt.axis('equal')
plt.show()

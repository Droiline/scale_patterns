import numpy as np
import math as m
import matplotlib
import matplotlib.pyplot as plt
#import timeit

from pattern import pattern_gen
from scalify import scalify
from mask import sml_mask, med_mask

# input = np.array([[0,0,0,0,0,0,0,0,0],
#                   [0,1,1,0,0,1,1,0,0],
#                   [0,1,0,1,1,1,0,1,1],
#                   [1,1,1,0,1,1,1,0,1],
#                   [0,0,0,0,0,0,0,0,0]])

iters = 14
show_all_iters = False
dims = (80,50)
# dims = (15,15)

pattern = pattern_gen(dims, iter_n=iters, ar=1, ir=2)

if show_all_iters:
    cols = m.ceil(m.sqrt(iters))
    rows = m.ceil(iters/cols)
    fig, ax = plt.subplots(rows, cols)
    fig.tight_layout()

    for i in range(iters):
        scales = scalify(pattern[i], med_mask)
        col = i//cols
        row = i%cols
        ax[col,row].pcolormesh(scales, cmap='YlOrBr')
        ax[col,row].axis('off')
        ax[col,row].axis('equal')
        ax[col,row].set_title(i)
else:
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()

    scales = scalify(pattern[-1], med_mask)
    ax.pcolormesh(scales, cmap='YlOrBr')
    ax.axis('off')
    ax.axis('equal')

plt.show()

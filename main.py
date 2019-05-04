#!/usr/bin/env python

import numpy as np
import math as m
import matplotlib
import matplotlib.pyplot as plt

from pattern import cell_automaton
from scalify import scalify
from mask import sml_mask, med_mask


run_type = 'ca'
show_all_iters = False
dims = (40, 36)
iters = 16


if run_type == 'ca':
    # 1: 6
    # 2: 18
    # 3: 36
    radius = 1
    lowers = list(range(6))
    uppers = list(range(6))
    # radius = 2
    # lowers = list(range(0, 18, 2))
    # uppers = list(range(0, 18, 2))

    fig, ax = plt.subplots(len(lowers), len(uppers))
    fig.tight_layout

    for y, lower in enumerate(lowers):
        for x, upper in enumerate(uppers):
            if lower <= upper:
                pattern = cell_automaton(dims, iters, radius, lower, upper)[-1]
                scales = scalify(pattern, sml_mask)

                ax[y, x].pcolormesh(scales, cmap='YlOrBr')
                ax[y, x].axis('off')
                ax[y, x].axis('equal')
                ax[y, x].set_title('lower: '+str(lower)+', upper: '+str(upper))


elif run_type == 'ah':
    ah_args = {'ar': 1,
               'ir':2,
               'x_sub_harms': [-0.48, -0.225, -0.5],
               'y_sub_harms': []}
    pattern = ah_with_substrate(shape, iters, **ah_args)

    if show_all_iters:
        cols = m.ceil(m.sqrt(iters))
        rows = m.ceil(iters/cols)
        fig, ax = plt.subplots(rows, cols)
        fig.tight_layout()

        for i in range(iters):
            scales = scalify(pattern[i], sml_mask)
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

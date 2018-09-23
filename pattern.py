import numpy as np
import math as m
import random as r

def pattern_gen(shape, iter_n=1, stoc=0.1):
    # Generate list of numbers between 0 and 1 that follow a cosine curve. Use these
    # as the initial probability of a scale being black.
    init_probs = [(m.cos(x*2*m.pi/(shape[1]-1))+1)/2 for x in range(shape[1])]
    # Use init_probs to generate the initial scale states
    iter_0 = np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in init_probs]).transpose()

    return iter_0

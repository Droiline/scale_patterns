import numpy as np
import math as m
import random as r

def init_even(x, shape):
    return m.cos(x*2*m.pi/(shape[1]-1))

def init_wideback(x, shape):
    return 0.96*(m.cos(x*2*m.pi/(shape[1]-1))-0.19*m.cos(x*4*m.pi/(shape[1]-1))-0.105*m.cos(x*6*m.pi/(shape[1]-1))-0.045*m.cos(x*8*m.pi/(shape[1]-1)))

def pattern_gen(shape, iter_n=2, stoc=0.1, ar=2, ir=4):
    # Generate list of numbers between 0 and 1 that follow a cosine curve. Use these
    # as inhibitor concentration values.
    inhib_c = [(init_even(x, shape)+1)/5 for x in range(shape[1])]
    # Use init_probs to generate the initial scale states
    #output = [np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in init_probs]).transpose())]
    # Pad out the array by the inhibitor radius on all sides
    back = np.zeros((shape[0]+2*ir, shape[1]+2*ir))
    front = np.zeros((shape[0]+2*ir, shape[1]+2*ir))
    front[ir:-ir,ir:-ir] = np.random.randint(2, size=shape)
    # The oscillator. Used to mirror the adding patterns.
    osc = 1

    for i in range(1, iter_n):
        back = front
        # Copy the horizontal boundaries into the padding, this will make the array wrap
        back[:,:ir] = back[:,-2*ir:-ir]
        back[:,-ir:] = back[:,ir:2*ir]
        for y in range(ir, shape[0]+ir):
            for x in range(ir, shape[1]+ir):
                ad = (np.sum(back[y-1:y+1, x-2*osc]) +
                      np.sum(back[y-2:y+2, x-1:x+1]) +
                             back[y, x+2*osc])
                id = (np.sum(back[y-1:y+1, x-4*osc]) +
                      np.sum(back[y-3:y+3, x-3*osc]) +
                      np.sum(back[y-4:y+4, x-2:x+2]) +
                      np.sum(back[y-2:y+2, x+3*osc]) +
                             back[y, x+4*osc])
                # print(ad, inhib_c[x-ir], id)
                if ad - inhib_c[x-ir]*id > 0:
                    front[y,x] = 0
                elif ad - inhib_c[x-ir]*id < 0:
                    front[y,x] = 1
                else:
                    front[y,x] = back[y,x]
            osc *= -1

    return back[ir:-ir,ir:-ir]

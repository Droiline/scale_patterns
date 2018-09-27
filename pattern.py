import numpy as np
import math as m
import random as r

def init_even(x, shape):
    return m.cos(x*2*m.pi/(shape[1]-1))

def init_wideback(x, shape):
    return 0.96*(m.cos(x*2*m.pi/(shape[1]-1))-0.19*m.cos(x*4*m.pi/(shape[1]-1))-0.105*m.cos(x*6*m.pi/(shape[1]-1))-0.045*m.cos(x*8*m.pi/(shape[1]-1)))

def compound_cosine(x, shape, *harmonic):
    compound = 0
    for h in harmonic:
        compound += h*m.cos(x*2*m.pi/(shape[1]-1))

def pattern_gen(shape, iter_n=2, stoc=0.1, ar=2, ir=4):
    # Generate list of numbers between 0 and 1 that follow a cosine curve. Use these
    # as inhibitor concentration values.
    inhib_c = [(init_even(x, shape)+1)/5 for x in range(shape[1])]
    #print(inhib_c)
    # Use init_probs to generate the initial scale states
    #output = [np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in init_probs]).transpose())]
    # Pad out the array by the inhibitor radius on all sides
    back = np.zeros((shape[0]+2*ir, shape[1]+2*ir))
    front = np.zeros((shape[0]+2*ir, shape[1]+2*ir))
    front[ir:-ir,ir:-ir] = np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in inhib_c]).transpose()
    # The oscillator. Used to mirror the adding patterns on odd lines.
    osc = 1
    output = [0]*iter_n
    output[0] = front[ir:-ir,ir:-ir].copy()

    for i in range(1, iter_n):
        back = front.copy()
        # Copy the horizontal boundaries into the padding, this will make the array wrap
        back[:,:ir] = back[:,-2*ir:-ir].copy()
        back[:,-ir:] = back[:,ir:2*ir].copy()
        for y in range(ir-1, shape[0]+ir):
            for x in range(ir-1, shape[1]+ir):
                ad = (np.sum(back[y-1:y+2, x-2*osc]) +
                      np.sum(back[y-2:y+3, x-1:x+2]) +
                             back[y, x+2*osc])
                id = (np.sum(back[y-1:y+2, x-4*osc]) +
                      np.sum(back[y-3:y+4, x-3*osc]) +
                      np.sum(back[y-4:y+5, x-2:x+3]) +
                      np.sum(back[y-2:y+3, x+3*osc]) +
                             back[y, x+4*osc])
                # print(ad, inhib_c[x-ir], id)
                if ad - inhib_c[x-ir]*id > 0:
                    # print("0")
                    front[y,x] = 0
                elif ad - inhib_c[x-ir]*id < 0:
                    # print("1")
                    front[y,x] = 1
                else:
                    # print("stay")
                    front[y,x] = back[y,x].copy()
            osc *= -1
        output[i] = front[ir:-ir,ir:-ir].copy()

    # output[0] = np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in inhib_c]).transpose()
    return output

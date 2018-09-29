import numpy as np
import math as m
import random as r

def init_even(x, shape):
    return m.cos(x*2*m.pi/(shape[1]-1))

def init_wideback(x, shape):
    return 0.96*(m.cos(x*2*m.pi/(shape[1]-1))-0.19*m.cos(x*4*m.pi/(shape[1]-1))-0.105*m.cos(x*6*m.pi/(shape[1]-1))-0.045*m.cos(x*8*m.pi/(shape[1]-1)))

def compound_cosine(x, shape, amplitude, harmonics):
    compound = 0
    for i, h in enumerate(harmonics):
        compound += h*m.cos(x*2*(i+1)*m.pi/(shape[1]-1))

    return amplitude*compound

# Calculates the sum of the diamond neighbours from the square slice given
def sum_neighbours(nbhood, osc):
    r = nbhood.shape[0]//2
    x = -r
    sum = 0
    for y in range(1, r+1, 2):
        sum += np.sum(nbhood[r-y:r+y+1, r+x*osc])
        x += 1
    sum += np.sum(nbhood[:, r-r//2:r+r//2+1])
    x = r
    for y in range(0, r, 2):
        sum += np.sum(nbhood[r-y:r+y+1, r+x*osc])
        x -= 1
    return sum

def pattern_gen(shape, iter_n=2, stoc=0.1, ar=2, ir=4):
    # Generate list of numbers between 0 and 1 that follow a cosine curve. Use these
    # as inhibitor concentration values.
    harms = [1]
    # harms = [1, -0.19, -0.105, 0.045]
    # harms = [1, -0.315, -0.055, 0.055]
    # inhib_c = [(compound_cosine(x, shape, 0.2, harms)+1)/5 for x in range(shape[1])]
    # inhib values from 0.23 - 0.45 are good
    inhib_c = [0.3]*shape[1]
    print(inhib_c)
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
        for y in range(ir, shape[0]+ir):
            for x in range(ir, shape[1]+ir):
                ad = sum_neighbours(back[y-ar:y+ar+1, x-ar:x+ar+1], osc)
                id = sum_neighbours(back[y-ir:y+ir+1, x-ir:x+ir+1], osc)
                # print(ad, inhib_c[x-ir], id)
                if ad - inhib_c[x-ir]*id > 0:
                    # print("0")
                    front[y,x] = 1
                elif ad - inhib_c[x-ir]*id < 0:
                    # print("1")
                    front[y,x] = 0
                else:
                    # print("stay")
                    front[y,x] = back[y,x].copy()
            osc *= -1
        output[i] = front[ir:-ir,ir:-ir].copy()

    # output[0] = np.array([r.choices([1,0], cum_weights=[prob,1], k=shape[0]) for prob in inhib_c]).transpose()
    return output

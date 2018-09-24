import numpy as np

def scalify(input, mask):
    input = np.array(input)

    # The output is the size of the input * the size of the masks. The scales
    # interleave heightwise, so only half the height is needed. An extra scale is
    # added on the side to deal with the edge scales, and then taken off before returning.
    output = np.tile(mask.lines, (int(input.shape[0]/2)+1, input.shape[1]+1))

    # The centres of the scales are offset from those on the line above and below
    # by half a scale's width. The offset variable will be added to the starting
    # position and negated at the end of each row.
    offset = -int(mask.lines.shape[1]/2)
    out_y = 0
    init_x = 0
    for y in range(input.shape[0]):
        out_x = init_x
        for x in range(input.shape[1]):
            if input[y,x]:
                output[out_y:out_y+mask.shape[0], out_x:out_x+mask.shape[1]] += mask.fill
            out_x += mask.shape[1]
        out_y += int(mask.shape[0]/2)
        offset *= -1
        init_x += offset

    # Edge Cases
    x = input.shape[1]-1
    out_x = x * mask.shape[1] + int(mask.shape[1]/2)
    out_y = int(mask.shape[1]/2)
    for y in range(1, input.shape[0],2):
        if input[y,x]:
            output[out_y:out_y+mask.shape[1], 0:int(mask.shape[1]/2)] += mask.fill[:, int(mask.shape[1]/2):]
        out_y += mask.shape[1]

    return output[:,:-mask.shape[1]]

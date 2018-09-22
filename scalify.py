#
import numpy as np
import mask

def scalify(input, m_size):
    input = np.array(input)

    if m_size == 8:
        lines = mask.lines8
        fill = mask.fill8
    else:
        lines = mask.lines16
        fill = mask.fill16

    output = np.tile(lines, (int(input.shape[0]/2)+1, input.shape[1]))
    out_i = 0
    for i in range(0, input.shape[0], 2):
        out_j = 0
        for j in range(input.shape[1]):
            if input[i,j]:
                output[out_i:out_i+m_size, out_j:out_j+m_size] += fill
            out_j += m_size
        out_i += m_size

    out_i = int(m_size/2)
    for i in range(1, input.shape[0], 2):
        out_j = int(m_size/2)
        for j in range(input.shape[1]-1):
            if input[i,j]:
                output[out_i:out_i+m_size, out_j:out_j+m_size] += fill
            out_j += m_size
        out_i += m_size

    # EDGE CASE TIME
    j = input.shape[0]-1
    out_j = j * m_size + int(m_size/2)
    print('out_j ', out_j)
    out_i = int(m_size/2)
    print('out_i ', out_i)
    for i in range(1, input.shape[1],2):
        print('input[', i, ', ', j, ']: ', input[i,j])
        if input[i,j]:
            print('Edge case detected')
            output[out_i:out_i+m_size, out_j:out_j+int(m_size/2)] += fill[:, 0:int(m_size/2)]
            output[out_i:out_i+m_size, 0:int(m_size/2)] += fill[:, int(m_size/2):]
        out_i += m_size
        print('out_i ', out_i)

    return output

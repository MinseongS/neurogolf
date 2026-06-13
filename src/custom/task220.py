"""task220 (ARC-AGI 913fb3ed): expand each pixel into a 3x3 halo block.

Generator rule: each input pixel at (r,c) with color g in {2,3,8} produces, in
the output, a 3x3 block centered at (r,c) filled with a "halo" color
h(g) (colormap 2->1, 3->6, 8->4), and the center cell keeps color g.
Pixels are placed with r,c in [1,size-2] (block never clipped) and the
generator guarantees the 3x3 blocks never overlap.

Because blocks never overlap, within any pixel's 3x3 neighborhood the only
nonzero input cell is the single center pixel. So per output color the rule is
purely linear over the input one-hot:

  halo channel h(g) = dilate3x3(In_g) - In_g   (ring cells, center cancels)
  center channel  g = In_g

This is a single 3x3 Conv mapping the 10 input one-hot channels to the 10
output one-hot channels -> one op straight into the free `output`, 0 memory.
"""

import numpy as np

from ..builders import conv_network


def build(task):
    # colormap: input color g -> halo color h(g)
    halo = {2: 1, 3: 6, 8: 4}

    # W[out_channel, in_channel, ky, kx], 3x3 kernel, center at (1,1).
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)

    ring = np.ones((3, 3), dtype=np.float32)
    ring[1, 1] = 0.0  # 8 surrounding cells (dilate minus center)

    for g, h in halo.items():
        # halo color channel: ring of the source-color input
        W[h, g] = ring
        # center keeps the original color
        W[g, g, 1, 1] = 1.0

    return conv_network(W, 3, 3)

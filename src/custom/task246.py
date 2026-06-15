"""Task 246 (ARC a2fd1cf0): red dot (2) at (r0,c0), green dot (3) at (r1,c1);
draw cyan (8) L-path: along row r0 from c0 (excl) to c1 (excl), then along
col c1 from r0 (incl) to r1 (excl).

See _hpwl.build_hpwl for the shared implementation (floor-break: fp16 MatMul
chain reduces R10 intermediate from 4800B f32 to 2400B fp16, memory 7440→5520B,
pts 16.02→16.29, FRESH 200/200).
"""

from ._hpwl import build_hpwl


def build(task):
    return build_hpwl(2, 3, 8)

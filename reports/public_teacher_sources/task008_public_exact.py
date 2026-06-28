"""Task 008 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task008.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJL8QGoC/5VWTW/bRhAVuaREjWNHXQep2sBOwCBFwZMiX4ieEgtFAboBCvtQoBeDWq5soaLIkJRt+McU/qmd2Q9Ztuim'
    'FjAid+e9t2/2Swrgl39ewVvw58ty1YBbj8GVGOkNZ2I8C/2zxVzITUCMgNgA4jUgvAeMgMmRUAj/diSKxVOYymKq4tpijoCG'
    '5awqx2H/VGYrIb+kN9EOeOmNrD+xO6cXvYTgbynLbJ7XQ+fOcRUpVqT4eSQaSbSP5P7nSKJ9pHbSAVA59BWDnhHuY8nlLGRI'
    'prlVLd05Db1JWjdRH9ymGPYNXxBfaD6iuI8iG3zV0p0t/I9qfO7Nzqtx2P1cXawtz+shWnYfWO4Q5RAUmvuzOZE2JbuU/1lL'
    '+nVJaTMTZ6t8W+kdaBDv1mW71kdVHtkT2/bY0/aEtifa7QljT/wfe8LYE+32cOHI3mn8nNk7jZU9JLVXrCQn8XMqnmjJSYvk'
    'AZj5BVOI2g3FVej9LusahqC9gF5Qzk7zKmRnqylm6N2yMbFoDOfNw0wvq86z4nqpaa91kjXXBfcxsypD9jnL8IyTAugusBzu'
    '4ctV6P95KStpzEy0GXTKJrlYm8H3dQlssmlmM9PLxPlCzhpN+0EnlZkAM9X84rLRfj4AicC6FywTLYnzS2vpEPR0gTIK3q2s'
    'Cu5m1XaeMqC4mBc2/6O+TNx8vH38vgcUMpI4L4tmZGraVwn3RND0lCfa8AFoDKg+QnCWVRd2IBITazHxUEwYMbEpJrSYUGIC'
    'xcRabB9ImvTrsHf2dSXlraSVnV1nlKnxAFU4b3qSCSwILNrAwoDNQn4AnArQdM7yqyrs/pY2OOiDkwM/AeU0Dq3l4+kWjo4D'
    '/XKou1EVHuNCyyXvYc80raUtJtTX4BRsQsO8Tcwh0Bjg4UaJQWW4t0inY5t/Y673PG65Rt8C9UNX0j5COhE5w2/LxisaW+CX'
    'aXY00qP72HE0CtkfaRbtg5cXmQwDUSzrJl02dw6jFVIQNH+ZLmveLVYN/kSG/q9fV+mCOxdRGLBB7xh/KZOh09Ef1zyZeVpM'
    'PU6GNvf4YzESMZY3ePRc68TJMPiWDmL639IZJcPOUzrvFYb+KtwXZgWcLVB1D3oMjvYG3WN1IhJPtXexTXdB4tFkRCfBgDpw'
    'nyafLN81s0cEH6OL0cOgqqkqwNjBeIGxi7GH8ZLEXqAUHrLEo9EjGLjHtM6JA9EOvqvtmTgd3aBdluBlvosNs2sSJ4i+BAHW'
    'pHeJdvScz96jZ3QQOAFgODiK3kQJdByXeX63F/T/sv/c+Gt4FTh8AG7gYADGIcX0HZg9pxD9bcSxB53Bd/8CgsMrPykKAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

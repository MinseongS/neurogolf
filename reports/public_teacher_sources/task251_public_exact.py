"""Task 251 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task251.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIACDpQGoC/72Uz2+bMBTHYwzBe+OQeluHJq3bOHJZQkgi7TBF6Q2tUqXedomcxN1QCUZAJv6M/Qk5bP/nbH400GrqxGG2'
    'rAfvfd/HBvuZwKffFnwAI4yTQw5aNgaNy8EKamyiu/WtY9xE4Za3JZ6UeJUk5buT5C1UKRRL4+iXLMvdZ6DlwtaOSFPhUk6x'
    'NI/DrwGLmIMK0mEsYiXCN4cNvAHjNmV7DrWX6rzIxw6+OkTwGcoXivfJ2DGvWHEtROS+AuuOpzGP1tl3lvAlXuIjMt0z0BO2'
    'y5ao6tIFNqjMDnnSJk8UedKbPOmQvTbZU2SvN9nrkKdt8lSRp73J0w7Zb5N9RfZ7k/0OedYmzxR51ps865DnbfJckee9yfMO'
    'edEmLxR50Yd8Xh91mU6NWOT7pDrpF81kUHkpCeOcp6FIq3nlimRlVXmmOORrVWdlxIfmHe5zqlIysj2LImd4KeIty93noLMi'
    'zGykSu4jVNFyhT/oUCJkhTv4mu3cF6DvxY47ZCviLGdxfkSYom+uQ/DIXMl7IrAHf2mNhksNqn3WA3vP8QJbe4ojNfgJDitO'
    'czW8Jse1CBppK/XPA4TcX4iobhFLOqurJfiJOu20hv/73GruF0Lkh5U7EywH/9jM2tIH9uu7+vqm5/CSIDoCjSA5QI4LNTbv'
    'od7+UqE9Vqx0GIzO/gACtkl6MAYAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

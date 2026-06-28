"""Task 348 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task348.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAKz8QGoC/71V207bQBD1JTH2JEAwCOWlgKwKUdOq4SJuVQUJQpUsIdryUvXF2tibxMLYqb0haZ9Q/6Jv/En5lH5Kd32J'
    '7SSo7UsdTSaeOWfn7OwlsnzyYxGOoex4/QEByeo1zJBAhfnAHzZM7IGMRjg0rd5QhXG4o5WvXcfCT1At351FjcJj6kGOesio'
    'kS+wqixCkOOaB6PDlPcCcjpymtpa6RyFRFdAIH5deeAF2IZcXbV6h1zHZuBwFngXCuVUKfrV0ZSP2B5Y+Hpwqy+CfINx33Zu'
    'wzrPOBcFLUuubyGXlTOdg31mmtQMupeOp1eghEZOWBcpa3qYPZimqvOFUEFvmZFeQRGhVrLXTgEe1diEct+nbYU8TFWY6yHW'
    'kPLFlwFyC+1tQ5ZP20cDDC1cBfAaCi2FAiLFdwP61dbEpmeDBklTYy2BWu0ht2MOHZv0qGbxetCG9Zk6S/YoBSxD9KJKthMS'
    'Fmy2Q9iHwlCQJNWq44WOjc0ADamIhXcBRgQHV0E8120o5CcmICe5RPzbiX7DHBn6UeOXLOzRUc0+ChzyNVot8dK32aJ3bn27'
    'zsXba4ERMwxM01QJ32EvW4s1GGuAJKWW2SZNJD3L5eM4Sx+x9Cc/gOdQWIMMzFCNFNWF+A2kbzjw/8HHFSGuqFb8AaGn2Twe'
    '7exo0rnvWYiMt320A08gjwGlj2yT+OZeQ5XiuCa+R7ZO15c2DWuy5XshQR554EW1TlB4s7d/ZJLAQV7XpSuGh7RtWN+Sxdpc'
    'a3xnGHWeix8h8WLi9c0ImVxURp174tG3I1z+DswGTX1lBji59TIwTJAyBYeRgrkJhcIMHBtPTuLKhNc/yDLFZY00zri/fFJJ'
    'K4lfTod8I/P0AzJf41vxITW2aPyR4+5/xpD7U/pFC51Ru6f2QO2R2q8z/TQi83IlJVvGyz+RaLzJcTVqG019idYtt9KDZQgC'
    'p++Oxyy3Jk6QsRpPZNL073wyi0pNaSXb1fC4//p8Xk/+39RVWJF5tQaCzFMDamvM2huQ7PsIoUwjWiXgavO/AZCQ/lWkBwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

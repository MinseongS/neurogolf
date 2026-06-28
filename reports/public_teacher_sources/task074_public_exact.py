"""Task 074 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task074.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEXpQGoC/2WSQU7CUBCG+15b2g6opQKCIpqaGNOVsDFxY4UYExI3hITEDXlCFSK02BbDkqNwDXcexSN4AROnZWDDa/9+'
    'zfT/p8306XD7pUIL1LE/m8fAepY2CCZB2H+1lVbgfzpFyL17oe9N+tFIzDyXuWzFNCcPykwMI1daH1iCGmyiFh9cY1xEsWMA'
    'j4MyXzEOl4Bli8dd2+iGwo9mQeSlfbxwij2YK7s86XOR+ECZjocLdHfszKOIR17oZEERi3G0blZMTdgM1cH31W35SSzSbH2b'
    'be1k5SSbT034GHONda6CpQYog5HwrUwwj3EWtvrwMRcTi705NzrTAcVM1mS99pWUruUdXlw8UUvUCvWN+kFJ94nD+WN6zdSa'
    '6fe0f5lEa3NzSqwST4jHxAqxTDwilohFYoF4SLSIeaJJPCDuE/eIOWKWCESDqBM1YoaoEhWiTOREp7odHG+mw22DxLisqBlN'
    'N57PaNdZJSjozDKB6wwFqFqil3Ogf5E6jF1HUwHJzP8DkStQNMQCAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

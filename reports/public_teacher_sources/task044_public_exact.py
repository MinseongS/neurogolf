"""Task 044 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task044.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    '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'
    'GVZ57MAsAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

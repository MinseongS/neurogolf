"""Task 201 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task201.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'XG7eg43J+FdCcBEyQB4AAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

"""Task 205 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task205.onnx` by `reports/scripts/public_teacher_extract.py`.
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
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

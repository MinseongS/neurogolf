"""Task 209 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task209.onnx` by `reports/scripts/public_teacher_extract.py`.
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

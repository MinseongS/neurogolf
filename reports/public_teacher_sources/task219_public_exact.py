"""Task 219 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task219.onnx` by `reports/scripts/public_teacher_extract.py`.
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

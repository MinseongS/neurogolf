"""Task 255 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task255.onnx` by `reports/scripts/public_teacher_extract.py`.
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

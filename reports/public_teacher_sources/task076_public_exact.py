"""Task 076 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task076.onnx` by `reports/scripts/public_teacher_extract.py`.
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

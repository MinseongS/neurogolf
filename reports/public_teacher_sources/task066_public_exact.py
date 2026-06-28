"""Task 066 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task066.onnx` by `reports/scripts/public_teacher_extract.py`.
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

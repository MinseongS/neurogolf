"""Task 002 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task002.onnx` by `reports/scripts/public_teacher_extract.py`.
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

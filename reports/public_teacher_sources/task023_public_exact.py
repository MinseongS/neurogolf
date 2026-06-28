"""Task 023 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task023.onnx` by `reports/scripts/public_teacher_extract.py`.
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

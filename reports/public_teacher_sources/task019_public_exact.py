"""Task 019 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task019.onnx` by `reports/scripts/public_teacher_extract.py`.
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

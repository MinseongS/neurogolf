"""Task 370 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task370.onnx` by `reports/scripts/public_teacher_extract.py`.
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

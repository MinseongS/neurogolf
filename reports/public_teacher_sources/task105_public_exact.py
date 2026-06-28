"""Task 105 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task105.onnx` by `reports/scripts/public_teacher_extract.py`.
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

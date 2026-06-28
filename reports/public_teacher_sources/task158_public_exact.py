"""Task 158 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task158.onnx` by `reports/scripts/public_teacher_extract.py`.
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

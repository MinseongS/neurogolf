"""Task 178 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task178.onnx` by `reports/scripts/public_teacher_extract.py`.
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

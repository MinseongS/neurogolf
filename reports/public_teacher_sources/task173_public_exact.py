"""Task 173 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task173.onnx` by `reports/scripts/public_teacher_extract.py`.
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

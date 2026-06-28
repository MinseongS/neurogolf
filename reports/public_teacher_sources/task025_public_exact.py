"""Task 025 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task025.onnx` by `reports/scripts/public_teacher_extract.py`.
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

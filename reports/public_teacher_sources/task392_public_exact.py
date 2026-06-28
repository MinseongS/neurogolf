"""Task 392 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task392.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'AAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

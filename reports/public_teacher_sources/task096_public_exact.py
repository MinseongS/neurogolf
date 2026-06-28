"""Task 096 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task096.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'jskxwgVKb/t/267L1TUvAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

"""Task 076 public-teacher exact source draft.

Generated from `public_candidates/lucifer_agi_circuit_20260628/extracted/task076.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'X+lBAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

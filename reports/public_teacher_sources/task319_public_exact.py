"""Task 319 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task319.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'WJj7P21J0+ZrZAAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

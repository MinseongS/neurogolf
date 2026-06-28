"""Task 328 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task328.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'YDzZMbtnrfQMUJubZZsz1/1/OXNxRm1XAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

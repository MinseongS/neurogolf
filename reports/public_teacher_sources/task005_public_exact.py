"""Task 005 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task005.onnx` by `reports/scripts/public_teacher_extract.py`.
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

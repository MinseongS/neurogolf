"""Task 198 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task198.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'XHnn/2JY4hHaFQAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

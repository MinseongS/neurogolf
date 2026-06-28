"""Task 324 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task324.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'XeHvrioo/RLttqHVO/sftK74T0ocAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

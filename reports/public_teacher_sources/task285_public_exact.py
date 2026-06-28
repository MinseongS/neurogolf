"""Task 285 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task285.onnx` by `reports/scripts/public_teacher_extract.py`.
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
    'rFpbjddRNIINBCNYwh/itEdP8hCXoKFIeZMUjXEpzGx5bhQNcqmFLfrrEWmUS9EQqGSXhrlUVC5fdcmyEeURjyzZajymsRs1'
    'Bs8Uwmm5CssNwSDmzpKG6Fi2c9kwplNiCsIp1ny8gcaSkrEdtlcZpDRgU5Jfr6hujaJbsQ9RNM0x/Y2J5pWm4Xcx1aVRHIbs'
    'r0AMxT3GBWk0UQuk+olwGLI9zjC1Bbbh64w4g+PRk4xo8aKq7NFfUcRq7pTfTMQWmZeeSHyZTpd1evFrsie/Jnvwq9iDX8We'
    '/Sr25FexB79KPfhV6tmvUk9+lXrwa6oHv6ZiS9sVc5GudsnUwjdXuw3XlXdYbhW1LpzCxadyi2108XHLfbOiMkKjgHqb7GuA'
    'pLHTeruramVxFRd7cUuqGUk1b97GaiNKFudG68WeihKVarsglSZIX80fSwr2a1Np/SFaNYUeKndfmo5fPGJru/FUpqE9cbeW'
    'xnQzbl476lW9Lu4GUZuVsly05VLQqrnNvI+TKnGU9iHy7Vw0lR6NtlXiWk3dde1Sr8hiN5c75RuxOK2ZNPQND/8fqKdd/ls5'
    'AAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

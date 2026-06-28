"""Task 202 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task202.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAFvoQGoC/41UwW7TQBD1ru3EmYJIV4CCoAF8obKExAUJISGSVKjFEhcOIHGpnPU2seLYYW0nUU/9lH4Kv8Af8Ccwa++m'
    'RkGoSZ61M/Pe7Mx6Nh68/enBCbhJtqpKoHzDOjyPxfmF75zk2Tp4AHcWQmYiPS/m0UqMyIhck25wCM4qiouR1XzRBe9AKxnI'
    'fMPzNJeYpfdZxBUXn6JtcABOtBXFyFYJ7oG3EGIVJ8tigBlpW47a/8jpP+UvoLUr88wau4iKMugBLfMB1cSb/Mwz633iEHZZ'
    'wL48r5gjV3nhd0+liEohVdyIdZz/FX8EtQBqN+sk2UwmsW+PsxiemF6V8AJjxaWQue9++F5FKTzXwtb2k4+nzJZ847tf50IK'
    'OAZlMVfyZZLtDinJ9s/l5U2amh5t22e6Rz+CJic0XFXaWsjSlBaAdrR6b+WfRlnMTY0+6L6g8TeH5M65SFPDeQ2Nzez59Mvt'
    'Z6UtO7v9jBztqlfbqccZo/OpKWYAaOhimbOM5MK0PYTarDswBJvPXhnlEPTrBeUFpxBZySifmfhjvFgz6KRiLVIchbwq8bLp'
    '5IzMgjce8QBB+mSCVzA8turP1Xt8jPCHuEJcI34gfiGssWX1x8FdVKgZCh0lCKBPlVmFRK9xcELCgwNc11WF5Hcw3O1GJ7qm'
    'ECxCbcftdL3et6f634A9hPseYX2gHkEAYqgwfQa6hZrR22dMHLD6h38A4FgUm1wEAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

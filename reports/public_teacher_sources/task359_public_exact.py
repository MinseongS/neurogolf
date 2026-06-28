"""Task 359 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task359.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAB/pQGoC/5VUTW/TQBDN+iubqRBmgao94CJzMz1AFaGIEzFCVJbKgR6QuFTO2GmtxnbZ2Krh1/A3uTH2bmyVBFAtrb1+'
    '8+bNm/EHh7e/OByDnRU3dQVm3HwRhkR/8jlNakzP6zx4CPw6TW+SLF8fsJ/MuMM+FQb+m30CpCccuYoXb6a+M5eXZ3ET7IEV'
    'N5mi7MxBysH75ByBriGs9upb7+N1FUzAqMoDUxNQE3AnYdo5tWUeN31Lfd10/Y7KjLfrTjuvNt4v6yWoOjSYPK7w6m8jHGky'
    'KjL+n/wCNAu0tOBYri5us2Ltjz/KNK5SCYfQg4LL8laFzU9lBR70gGC4PSUBVlmkM2AojAJ987xewFPoZt5itiT93DfP6hXs'
    'QzdpIB4NaMA93Tw4P1JZXizpiRVlMbjzdL9DHP+IdwlUi86zbYsUxy6Ou+OHoFyCs8yW1feZMOX8lW/Ok4Qst3tQyi3+etNK'
    'u+8TDDlXnZMU3pXCcJCiPSgTLT5I0X6QwnAzRFKllykUTh6vr9NEyXigb0npKksa4ZR1RV+fb3/4VscrwS6Dx5y547D9diNu'
    'jtQxgKcRNzbgnmuG3eOLGAtcl4V6wpHVhR9QWPuK2EnwjDMOtBjBqngEI2aYlu2M+eTrkf4PiH14wplwweCMFtDy2rV4Dtpr'
    'x5hsM0ILRu6j33Myc1SEBAAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

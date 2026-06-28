"""Task 355 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task355.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAK38QGoC/5VTbWvbMBD2W2z51nWZMrp0ZO1w2Qr+tDQUSj+lGWNgVigrY7AvmRKL1Glip5a9mf6a/JqxnzXJltI0TTpm'
    'kI+7557T6aQHIayd/gFoQy2KZ3kGFimOOtgeJnmcMc/9QsN8SC/zqf8M0DWlszCasqY21w04kRRsp8mv/rRQyeek8J+IOpR1'
    'zbnuPMYcJpMNTGMt8wjkZmDyPrEjnGGcPdon51TbSI5w/sV5B6o0qHxsDQZJ8coR/2k+8czzfAItKKMg54WdacRYFI888zIf'
    'wL6Kg5ldpdj5SSZR2B94zqeUkoymcAgqBooJZkxH2GHDJKXt917t2xVNKZyCimCX95Ok/SgsPPssHS1mFrGmzlt/eJa3ACQl'
    '8YiT+xHc0XGN0QnvpvbxJicT3mzll+F26FkfCMt8F4wsaRqizgFUCDh5zG76pMC2IOQnnvuVB3JKbynsgQxik9t7RUR30AER'
    'B2tGQgbWLU0TbCd5xh+DZ16Q0G+ANU1C6qFhErOMxNlcN7GTEXbdOT72W8ioO73yiQZ1Q6s+U1q/gXSOiksOkAL9p3W9J6Yf'
    'WJr246xy+YS522399gcIOGVpPsGFJGq6tKvbWNLWpLWldaRF0rqqg0Nk8j3U0IKmtqGw/xkhcToxmqCr/ef3esX62/yk5YCD'
    'suPv+0p0O/AC6bgOBtL5Ar72xBq8AXkTmzLGzcU734YtnoFUxrildIkx1DmytcwVaKXAtejuQmsPyu7eqW8V2qmUt44ilbQO'
    'kmorIfc+pPS1ynq5LBkAhBxsCXDcUIIRQbcMQs8Crf78Lx3UPepVBQAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

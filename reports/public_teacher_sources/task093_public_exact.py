"""Task 093 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task093.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJv8QGoC/4WVWXPbNhDHsdRham2zCpJJZMeTdtJ2nOF02k76lulMZfqKOZbv+40Wad3UQSmW/eSPkn7SdgFaCiUBiixY'
    'xP7+u8BiQcBEzlbZe/aBfWSf/l3Fd5iphZ1BH6GN0EHocvhCPHParJWDjyzBewiR5PdqTt+B5MMkf4vwhcMDmdKbXtS3c2j0'
    '2wXjKxgE/0S45/BIMLvRq5S8ob2IaW9Yi6TA/gHNRhB0/ForKrCxx5DDhsIjpfH4S47hkEfuJPAH5WDsFERFGmZB40TDbKqd'
    'UhonStXhsDWVam4MNzlsa+Ejh50piGO4wWFXDTnCDkKZwx7x1Ibvx7ZdaXO/2d4gbCHsIdxy2Bf1uawGvSAG2wiuBKUpEHt4'
    'HA6UHgQOk+Algo+wz+FIzHU/iKLYeEB2DseTxgChxOFk0nhIdg6nCSMlf8ThTL19CB5zONfCEw4XWnjK4VINf0c4Q7hDqCA8'
    'jB7Efw5X5LF8vF8LA69X8vqlQTPWn2v01xr9WHYxqb9R6f9I6C8n9IbnqRwKsnT1WHGbLFFB1u6ZlJNkhdYSoYnkQMifQkcI'
    'DUI+oSCJfiFjQMY78aach1F3EASPwcSbQqqfRRENrzJXtEKhKK8rhGvSVpOjvCZUpXZHoJbYHOuixIZXF3HPel4YddpRQG9m'
    'uhP0WkVWNIoQx14X5Ta8xneFqzRKI16GMjk0k7MQrB6vQ5NYa3odWmQM1SkacXShqpOqPVclxqGz+IZ+PBJ3pleiQy0k0E2s'
    'xCuy1ah1yd4ju3HYe15SOrWrcbmjZKDfCNFpHiLUCPXFQXHk+fZLTLfafvDeLLfDqO+F/a+Qep4SHe8tnm0P+nTYi0jb3YFH'
    'm41Dxc7l0QHPNZ5c28yjDcyB29hYdg22Zf9qWqLju2uMsb9ZkTlsi22zHbbLPj99ZntPe8wl33XTMkEIg+8IF/PgwJ2bZuzp'
    'HxrRcKDiAoufqi5k4qeaCxg/1V34z35rAv1Zot9wLTBS6Ux2wczh4tKylYRN17KWlxYxZy5kM+mUAfYKIRQCgVsusrGz/cI0'
    '8wufTCY/+bwDob1kpsiUor4D7VEPLMuBzrhnpBzojnqZNCl7o16WLhGIxoylHeiPGdDiDE7YzY/Ply5/ja9M4Hk0TKCG1N6J'
    'dvsTPldKKnKzirq8lyfdRbNEE/BeAbPiV8ChAkqBgA8SGoqwb8Q1xzFvLvClpKcAGzrgSACzYFMHtiTIzYJtHdiRAGfBrg7s'
    '6YCrA/s6UNKBAx041IEjRYJy2Y914GQKWKNQpzpwJoExG+pcBy6mwDjUpRLQFrpSbCEY7czrefBGs/mgvibPUx0tyItvNoGY'
    'lLXEVyZdkNeiiqzJa2zeHCtTr+wkrWpfLkFrGt94RnXlJhCkoSVNbX4tbX6htkCCthVz/EY7c/PravMTtDeXRnMj9xVUHpRO'
    'Gll+6X8tF9VeNg0AAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

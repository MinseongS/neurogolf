"""Task 014 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task014.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEjpQGoC/61V227bRhAll1dNbtI6viTIpSCCFOFDYcYvTl6a0A2SEDESOEUD5EVYLdcWU4mUuSQs9Kmf4k/sJ3R2Scpy'
    'JBcpYAoDSnPOmZ3ZnR358PKfPkTgZPmsrsBm8+d71OVFnVcy6B2JtObicz0N74D/pxCzNJvKHfPcJPAYWhY4RS6Gx9RnIyny'
    'ajgK7A9CSngACw84fLw7HFFnxFLEyccSAmh+gDMT+fC4C0Ytln4LnC9jUQr4BdQvasaB+7o8Oczy8IZKMGsyWE3pHpgxteJ6'
    'P7APmKzCHpCq2CEK2kQIrFJGlMRR4B0JOWYzAT93daOXOrMJy0XgvmUVLn9pLSymQSmZjS5F7yn0RYd6ZXE2HLPFzh2yeRNH'
    'yFfWuemt5nwh5cXkKilZK92DbjlKyt1mjzoR5k3W7hGK2oUo4asia63ocHml6MdWCndgIMVE8Go4we0aZnkq5otwFzlEP5bD'
    'f4S7D1g+tcrd7NLJuC3GEeNXYGWEuugKHWJ8HXYXlAbUgtQaT6PA+lyPYBvU9+YyZNR+NxF5YL1OU0Xnis4V/WyJfrZE/7Kg'
    'b+vAQOSpKkle9Oq2DtEAfBnYAjtL56XSSWqXJTtrIjV+rmTo5wv/fdAkPIRJNpuyOWqQF1h4wRTGL2N8gT0BbH7QZGrJcbly'
    'U/RNewYKA62jnhxnx5VIV6jqjLEileAR6M3CarO8HR0NcAB6W7DaBbAJigXKg1ljM2BFeQpPoVsI1PUH7y9RFsN6n7pZLrNU'
    'dAPlKWgRtG5wi7pSNKyWSRntd7yX0HnAnrFUfkfc2w2sTywNN8CeFhjd50UuK5ZX56aFcxGHHWYPHZcqMc6YwHlzWrMJNU/C'
    'Bz7pe7GetUmfGM1jte9w2zebT9+Mm9ma2Ibx96/hQ3SCBnpxM1ETMA1jrU5PVdQ9qd6HG+jyYjX9Et9cZbvNKlliKzCkmo1t'
    'tkT+5A+Qptss+U15FELapDE5w0Fz0Tw0H62HBmg30G6i3UK7jXYHra8i/u7TNiJP3l1HxIGKuqHL6Zo3sR8q5x+6zoGGdMNd'
    'Uwlf2x3sCjm4xkK2FqdD4q6bE9MIN5f8bVMmJoQffF81lOrV5JXxP59b7ftmd9oXfUbippkTMExi2Y7r+b2vj9s/TboFd32T'
    '9oH4JhqgPVI2+gnalteM3iojtsHoD/4Fx/JBansIAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

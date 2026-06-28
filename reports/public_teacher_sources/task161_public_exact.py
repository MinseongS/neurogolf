"""Task 161 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task161.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAFnoQGoC/4VUzW7TQBD2+i/OFETYVlGE+oN8QMIqkp3EFy6liVAlCy7pAYkLmGRprdhJ8cahj8Ej8Ci8CM8Csz9uGpoE'
    'S+vZme+b2Z3Z2fXg9e9HcApONrupFuCO53n4iWvJtEypLaTvXObZmN1jl/Pvki0l0xLZQtbsaMVOb8Nuj9o8zXO/OWKTaswu'
    'qyJ4At6UsZtJVvCO8ZOYcAKSAxaPuPgxsNCVEl7HfAVyQ+BgSIxojsOd8ZAudnRHL3fT20Bw2a+9mFpXvdhvXJQsXbAS9pW9'
    'j/a8H/v2O8Y5HIAggbBQK+N93zqfTZCKewJciNozlpZonEzgEKSCIaKYumI6n66it0G4g7ZTs5iqUM8Ap9QqpplvD1O+CJpg'
    'LuYdV+w0BGFHcHztu+fl1fv0NtgDO73NeIcg4WFuL3XlhAttYFWKlE999yJdXLNyzVdQZdUUFb22U7v1GYsKRz0qGoFXxbYq'
    'b/DpUtFq//M5Ah0ZaxjG1EHlfgmPQAfRMCrrcJ2wwkV6xRqsk9Sw0FbwoSqEjb9q7SRM1TMSALOKcN1lmlfqzE9kEQtQNoRD'
    '6ozybMZ85wOWkQmCWGeNMFwnqDRB+YFddeOYmqPlPYJMFJRfTRjeEbAZR0tAg0hpwnwL2wSzlQq4OVuynFN3Xi3wNHzn7bcq'
    'zSm5Cl54Vqsx0Lc26Rj6M7W0tAyOPBN56m4lrR1whDDZAXdXcB0lOJawfpZWuLEJZ0kLtJ1swtMNy2tcPWTb46sHbnt89fBt'
    '2P++RxAXD1nikQdGlnjwjxGrnXh3yz9ukYF4iRLbMD6/0Wpfqj9qNZLqrzOthlI1zoI9DGgOsCETQoKmnIYJMYI9nMoeScgf'
    '3D/xAIeg6lZIwCCmZTtuw2t+PNG3lLbhwCO0BaZHcACOYzG+PAfdOZLRfMgY2GC0nv4FLvyYSGgGAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

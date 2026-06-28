"""Task 004 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task004.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIADbpQGoC/71Uy27TQBT12E48vglpOjxUWLTI4iG8gBapEoJNSIUqWa2E2IBgEU3iaWPVtYNnnKa8f4B/KJ/AN/BhMI5v'
    '8zDZsMGRdXzuK+feuTaFpz+a8A5qUTLKFZivB2D2B6w+SEPRO/LsvTQZ+wzcMIq5itJEdlqd1gVx/OvQPBFZIuKeHPKR6Jgd'
    'szCvgz3ioewY5U+bYAuwGrML1DW5VL4Lpko3dI4JN2HqADt/vLvLamMeR6FnHwgpYRNKihFmvs1sbZBe7fVQZAJuwZRqxw5z'
    'eXLeG6RxmnnWYZTADswtYPejcMLcvojTs542e/V9rnQJvwE2n0SyVLIqJYuOh2plilWkPCsVMMh04SGXvfyJ574SYT4Qh3xS'
    'xgrZsYrhrAE9EWIURqdygxTJ92AhjTn4vDQht4h7CJc+VEULmoiJWt3HA5h3CvMOGBVJOEqjRHnOfia4EhncgVktmLmZe5qO'
    'hZ5DEnrWGz2L+zC3zEa+jWGLB7IUqENwOq5U/Hwp8BHMk8GWRVOOHEZHSoSrB30X5kXgMpSBHPCYZ700V/rU+USPCjdmwYOb'
    '1URLuWCo4zksmcv1xYQGeo7yOPaslzz0r4J9Wuww1f1pNYm6IJZWthgIzojHQinB6vqv9Uvl1V68z3nMyLH/k1BCgZrUbJOu'
    'fteCC2IYX38ZS9eXCv9c4Z8q/GOFf6jw8wqfVPhZhY+XuN+ipBDbHwS21rrnN9pmdzqegPz2XU30KQfEwMedgBD/O6HtttOd'
    'rmrwjRAsZSJaiDZiDbGO6CBSRBcREBuITcQriC3EteV+jJkeWeqp+v+3Pv+A0kJOsWlBx/jHq1WttjXdKb1Z+gAudy8Ag5iW'
    'Xas71H27hV93dgOuUcLaYFKib9D3ZnH3bwOu6jTC/Tuia4PRXv8Dd4Q3VSwGAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

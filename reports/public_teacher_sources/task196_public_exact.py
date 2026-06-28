"""Task 196 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task196.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIADzpQGoC/62T32rUQBTG50+yyR4LptFKKWKXICJDEXdTpPWmS3oXUmj1QvBmmWymbWhM1kxipVd9Ap9h37ROMllJtb0Q'
    'dobDmTP5OL/JMJ8NH38N4TWYab6oKxjEsuJlBUYs8sQdxFktZuee+TlL5wJ2odtwjSZ7xjGXFRsCqYptssQEImg/APkugd7U'
    'B2BKIZLrtibXN3/2XRrzxHtyFqW54OVxkf9gm2AseCKnWM8ltuAMGplrLooie+9ZJ/znqVqxLdi4EmUuspm85AsxpVOq1A80'
    'YA5YsirTRMhVy23Q3fQ51W/wZOzRkzSHT9AWmjZeK23co036tImmTdZKm/Rofp/ma5q/Vprfo+33afuaFq2HtqNp3ftyzYtS'
    'iNwzIiEl+KDLts+FC20xO6+zzKOnPGHPwPhWJMKz50WuHndeLTGFN9DTAZn7nQHcQVFXKnvml0tRCtequLwaH35g72zDsYLO'
    'HuEIdQOjhwfba/WtjcLRSkW6/PSvzDYcHChzhAZCt0cMHBI0NgkxYkOHBso6zfKljdWkNlVb2ljhEN2hO6ICsci2G2BzB+EU'
    '/ef450AHigQNrznY3A/f3tffHj3W6evu6iZfwHMbuw4QG6sAFa+aiEfQ3fFjisAA5Gz+BgmgCe+ZBAAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

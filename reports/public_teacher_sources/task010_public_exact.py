"""Task 010 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task010.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEXpQGoC/81UTU/bQBDd569shlYKU6AIqUhFPbQ+cqnKpeAAgaiHClRV6s0kCxgSO4ltvn5Nf0B/Qi/9Ze14U0cIpRx6'
    'qLrWPnnfvJm3s15ZE6s1taFeq0219YPoJflJOioLQkw4IfQIfcZAJP7xIOmZTUVvCQPGUKjmkemXPXNcDsOn5MU3Jt92LtRX'
    'NMJF0pfGjPrJMF9VFeVI4gvCkJFKoteO8yJcIKfIVp174ZSRPQgHdfiVDY8q209pPi6NuTMzW9faimpFVIQRYyzCRmdi4sJM'
    'psXHjMmfiktPE0Y+vyf3kZ6WCIaQM4rqiPbGZTwQ9jmhsBs5Y5RV4PO5mZhpE6eEjHElbNCJC+GnPsm9oqK6YlzPb9WpW31P'
    'KBk3jFvGHWPHfqPRIClmBe3GnXCJ/LzitzF96gLvCAnhxuKtxTuLO4KMqNphO0t78cOCU+9rRpuxy9hj7P+N9zmhbXHX4p7F'
    'fUFG5zFvOZ0OIWIczFGhVr0hHBAuGIeicj/G/XCZvGHWNxu6l6V5EadFJXVFukq4JBxykJWFXP3qjnwweb6pGGfhE+22Gluu'
    'rxAhrldBU0c4qVdw3Ai92QpOhH74TEN7MtEKQk9ZiRGSWhI97VJTNwLfcx2okEQj5FlX3ltVhs0S5ryL5r06wiRdT8kIF7UW'
    'K129K18tR7gIvzsiIr0+FV52vzmVqecHDf1z3pj5/+Oo+i/HkfpS//l4hZY0uEVyoDJJ5no119TJBv2+IlbTnKeJPFKthV9d'
    'jgUNVgUAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

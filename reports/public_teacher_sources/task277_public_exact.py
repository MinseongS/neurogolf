"""Task 277 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task277.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAFvoQGoC/7VVQW/TMBSu7TRxnkB0ZkOFslHlhHJam3ZMnNrALpEGFbtxmdImsG5pU+ZE3c/ZP+HKn+C/YDdO13WrxaFz'
    '9GT7fe/zlzjPzxQ+/mbwDqrj6SzPAI+4sFhYyMjo4tipniVjMX8NcsbQwDE+hTxzbcBZWse3CMMemH3/7OTkM6ABI33/0CGn'
    'eQJfQY4ZmcwOHes0vBmkaeLuwbOr+HoaJ+f8IpzFPdIjt8hyd8CYhRHvoeKRrhpYPLseRzFXHmAg11IirRWRlhRpbVGkpUTa'
    'KyJtKdLeokhbiXgrIp4U8bYo4imRzopIR4p0tijSUSLdFZGuFOluUaSrRI4KkYYUOQLMQ2FDaQz1y2RdghKIC9AvwR1AfUA+'
    'M+Y8nzikH0XwHhYTZvBJeOPY3+IoH8Xizd0XQK/ieBaNJ7yOZK43ikhYRDJzzOfiRZ3qya88TGAflIPhef7wnNTBzObpeX4M'
    'AmZVMZ5wh5zlQ9gV3waFg5EkHBaf+AbkGKxRmqTXXKRGeuGVUh9AzsAUm5eKU2vOfoQJj5kpJuIUO2QQRu5LMCapeDs6Sqc8'
    'C6fZLSIM/XRdatQsXxz0oFlRjVYeb8vYOGgi5bNVD2v9Mja8W7fkYNWTMvYvoohiChRq2FflI/iDECZG1bSoDU83AptaZtUg'
    'GD3dyGUUyb3gYbDc2qVvGFC07osDWu6R+1xuSZEpAcJuXewUEYaEu8yFgFQE8oVSQVY5EPQq/9ks1e+u9ULY9lUmBajyvbwT'
    '2CvYpYjVAFMkDIQdSBs2QeXbIsJ+GHG5X9wa9xcoQ+CyIc+0BPEj4H5xf9yH8SosbgMdLC8GLbulZ7f17Lae7enZnp7d0bM7'
    'enZXz+7q2UcbYfG/+hv/V0MW1U3ggaqwGnxRUR/iC7tsLkvr/VS7W+HtoqpuWv9dWV412SaKrQ4W9XZNnJSwb0CltvMPlygd'
    '20oJAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

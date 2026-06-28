"""Task 335 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task335.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAFvoQGoC/+1X3YrTQBRu0qSdnq5ud1Z0b6xLFJXc2O2iiAjuVkQIFkUvBG9KNhnaYJqJSbrteuWj9BEEr3wFn8qZZKZJ'
    'f1YoXshipgzfZM75zvkymZxMETz7cQhd0L0gnCTQjOh0MCXecJTEGPGLkFLf0F7S4NzcAX0Y0Ul4oM4VtcBxqJ9z+MWlnCew'
    'iIk1NoqN2mk07NszswmaPfPi1M3cBfSZkND1xvFBRfBkXKyx0TqvupH3GNIsLNfYC4zGe+JOHNL3goxG4hNlrtSXaMoKzZ4t'
    'aDLbn2hcG1O4bTZJ2ybb7UwkqF6H9SPesRp1Df2D7zmEm3lUYepmZuepND8A4E/CoTRyY0jXB6czw2TA1ddfR8ROSAT3Vh3t'
    'WeboM0emV3tD4liGy8hQsGebiIYkMKqngQt3l8IxvVkwLx5ExDX0V18mts+j8cctczqpOD6zQdyyIxfHZ1bF5WQo2LPdmotb'
    'DsdWC6d7m4lzLuwgV7egQdGOYUQj7+sg3Z/q2wgMWNz8sqN2TqJkkTK/fyhEwHo6zrxuQUqBbA5roZ2M0hSPIB2DHtrucQfX'
    '+MVxx6i+s11zH7QxdYnB1AZxYgfJXKnCfRA+ULsgvk+n4i3GNTpJGBr6xxGJCFaG5q8GUpGOFNRuKb1iYbC+Nyr/Xfv24t/0'
    'spWt3M9/22QxayOFF7PCiaUsZmUxK1u5n69SMcOsitV77OBvobW5Iwspq3NdC6ly7qeC+O9Gaiqcxa25pFXkQHKqAjWBusCa'
    'wLpAKUYWVBDYFLgj8JrA6wJ3BbYE7gnEAvdXtDP1XHt+VL8K2vsIMdHZOd062faRwwqaz9kqAF8L9jkTJ3nr4eYXY719uiNP'
    '/TeBbQTcAhUprAPrbd7PDkH8H7jMo6dBpbX3GyL8azk+EAAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

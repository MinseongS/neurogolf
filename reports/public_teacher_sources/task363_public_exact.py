"""Task 363 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task363.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIADXpQGoC/61WS2/bRhAW38w0QJV17DpBm9pskwODtJIdO057UWwURYSmKGKgh14Eilw5jCSS5cN1/Ut69P/sIZ19URIp'
    'tQ7QFehZznzfzHB3Z9YufPfXLjwDK06yqgSnGI1nQTgFh8qJHlwRh09HE886n8UhBV/BbQavKNhUSAa22Ww99iIP/mRYLjmW'
    'zRbYfVCRiMUnnnkWFKV/B/Qy3YUbTYevFxBXTKqTFZTOUB7ILFjqKDdgRHTicLkO8z0oPnHy9I8sp4V35y2NqpC+Ca78T8AM'
    'rmgxMG40x/8U3CmlWRTPi91Okxyms01kfS35EFRAouc9z36VX9SkuODZrSXJQEQPb0v6DDAAGHHxjhh5L/ect7R4F2SUGUJl'
    'CJcNR2Bd0zztK8FowCDEnhZlkJeefZYmYVDWkXmghyDNYE2LLEiIOaVJ5BmvoghO6rUCc5wFERjXuGwu1+GrZ/wSRP4WmPM0'
    'op4bpgk6SsobzYCnUKNq/9yxOGBTmrOdlQfsEUgFMVC2Txd+C+o5EwpKo1GYVkmpdu28mq8sn8YoZyCOKmcSqwjTnKLjNLn0'
    't+Eu6hI6G/F1G5gDk+31PTAx2WLQwZ/Bzw58BYIIS1GJcxnM4mg09qwffq+CGasOqSEWn7TzfyLznxT48HIEY1KgK9ROZnGm'
    '1uE1CA+gDMTJgjgpsRRvlTqmzdJnqWNWkgrm9aR/rDyNPefHnAYlplNDMHE+aRfartoY/ICyR/Skp776AeALKvpt0gs09dF0'
    'UJdVnLAUeVl1BtpAb1em1orGPBwuRztExfP10Z6j6ehjo+2IAjEn8SVF/vFyrGNUvGjH2uHlBGaa0NeIOFmmnKDiZZuyhSaW'
    '4ktiZmnR94w31Ywrj4AriD6Rym1UHmBO8x6x8NBhd1HqCavleV+oJfoBCJAQfWIyIYr2IYjtBK4jNn858IzzagyfLwpa6omZ'
    'VsyKHQm+gLp5r5h7gvx4yVzPCLBuM8pmQUIF7FfgHFgycM3BimZ5rjo9sYOwxO1oNSpN7LQ0g43e6m7EX6pycy8i2oXvu2bX'
    'OcXCG+515NCk1KU0pPTvduGU18wQTf63nKku4AV901AEqghaIx40pP8NJ8g7exFA3xRA4qnEa438N/tnq7zwb/2nf4FX/u1N'
    '/rdcDfHsQhq6Cuxvc6W4iYZu7fsZ9y0umvbSmA3p/+S6COdXz3DQ+chhNDcWuvopOzJDraMOxKRo72iLp7C0nfHfH8RQ0n/K'
    'say7t8EfGsPfdzX8mUjBxLDBDrsMq4k/InYD0h92NRwr+3WfLzXvY0PXamhZq1raloq7A5cvBbaaYXSbldQaU22t6V9qohG2'
    '3wjb/KT/afg/8+Mju8XtD5A6fvcb8rcv5T/NZAdweUkXdFfDB/B5xJ7xHshuxBF6G/F+r/5XZxXBHpM97/cXtz+DQBtyakKn'
    'e+8fRI5dpiMMAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

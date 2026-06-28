"""Task 215 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task215.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAKP8QGoC/71TTW/aQBDF38s0aZ0tqShSILF6qaVKYNND20tED5Es9ZIcKvWCFtgWQwKINYp770/oD8hP7Xi9ayySXmNp'
    '1rvvPc/OeucR8vkvwAU46Wqzy8AUfYwBRkSdaX8cxYFzc5tOOXyAcg0Wy2NqTVdZ0Lzms92U3+zuwldAlpxvZumdaDceDBN6'
    'UEhwiFgxTHBgOTWnkc63F5ScFlQbdgDVGDH15kz8/DWeBN7VlrOMb+EcNCYTaIUIvGsu5mzD4Z1WCLA3LIvlOKSu4Lc4CZzv'
    'c77l8EWdm3rb9T1b/R7qM31jeXgMNsu5uDQurQfDe3xE3EJ9RUk5wRLtr0xkYRPMbN1uFqoAKhIcMd/EfeqWwL7YC1AQqPrA'
    'WkafqJXOcl3pe31DBUjd9S7DReBesQzp8EVRairaJu5IIWNiGQ0+ju+jsENM3xvhrSZ+Qz2melfcIPENhdmHXLTnqu9eEwO5'
    'og0SYh2AeN0JaTwCJwkxDkGWJ6TK+ccgXd8dyatKckuV4mA857xexjDJ3Rr1nPPwVP6islsS0tU/6RhLKxojsc9w+aOnm/cN'
    'tIhBfTCJgQEY3SIm56D65H+KRU+Z+kBQhCkFb6VLKQUf6aM6vWgX/nyCMSQTP8mcVb6VdPOAPq1MSwEI0raEW9oWEnUlWpRW'
    '2e8lHOFWROXpLjp7z0muWeNa2mu1LbqLk9JXtfwjGxr+yT+Wzak+HwUAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

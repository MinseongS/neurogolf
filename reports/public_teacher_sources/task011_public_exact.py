"""Task 011 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task011.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJP8QGoC/6WX+Y7bNhCHLVu26XGCOkqbukLabN0DgYCi1uEDBYrugaKJgB5okKLIP4JWZrre9UqOJCeLPM2iz9AHLEVR'
    'EnXQi8AGaHLI4Y/fSEPSRqB8FrvR1VTXnSgOd168C92Ng2+2QRj/8N/X8Ay6a3+7iwFFsRvGkTOFHvZXSY3cGxw53sU75b4f'
    'rCPsnG8C78qZqmVz0n2xWXsY/oJyvzJMzWh3TebwxmTwJ17tPPxid619BHKyzLF03D7u3Ep90oGuMN6u1tfRuHUrteuEOiPU'
    'RYR6mVAXEOo8oc4T6gcSGozQEBEaZUJDQGjwhAZPaBxIaDJCU0RolglNAaHJE5o8oXkgocUILRGhVSa0BIQWT2jxhNaBhDNG'
    'OBMRzsqEMwHhjCec8YSzAwnnjHAuIpyXCecCwjlPOOcJ5wcSLhjhQkS4KBMuBIQLnnDBEy4OJFwywqWIcFkmXAoIlzzhkidc'
    'fhjhvxLwhylv6Lxh8IbJGxZvzHhjzhsL3lgqkBuRyrUnvbPA99xYGybkawZpAucCvfc4DJzXCngb7PrOeRBsVK496f78Zudu'
    'YAFcpzJM2683gRurvDGRz9wo1gbQjoOxlKz2HPhxZZAa69WNWjQnvZPwn1/dmxJn/elaAGHwzklfPhTTlXY4VUmZ9H5x4wsc'
    'lsMls7xg0zTLI7M8wazHQARhEF+EGDvruUXW0Mka+qRzslolo1551COjHht9msxV5HDqvFXp92Tw0o/e7DB+j9NVSBKRVfqJ'
    'p0c8Perp3eEZ6kRTp5r6HZrE06Oe3j7P5+zlv81rSguURAG2ySJMEqJoN+fU79ANfEzmPiAu2IvJVnR9H28iKqpTUfKtILpf'
    'E8m81Sz4Mtvr3NKQz4EB3e8+2fCJA96mbWWYro5XTnCh8ka28c+A71V6JDOC0FRZXctDqTEPvwfmT7KZ1s5uqRbN0h5opwdX'
    'MQqDrbtyqAm9CG9JF8OwGIY16fzhrrSHIF8HKzxBXuCTZ+DHt1IHfmJLk5s3Xm8wzWOgPQ7ZGmTzF+1aXtPt+CNwLpxIn/bq'
    'upo1atNpJL9BNg7DPA5zChDs4mi9wkk0fdapZo098XwHmRM5JtKUITiR0iN65O2rrGankDLOfi+zs4ie3kkM2lPUGfVP82vA'
    'Hkut9NNmdYfVmo5k4lkkkH3ERlrCKQadwiWafSRV5lRrbTSSTtnZasu05xUaJCrFEWY/awk+sqBGgjrTLg66Qrs690P7tUdI'
    'ItrsiLBzd+0T2p/uexvlgX9Ju+vngI36mcunyQvID08b5Q/6BPWSoTyz7Gnrjk/tud8ftU/ZvrKlrnaBhkleZHlu/y0S6tzx'
    'QjuV5KjW2ldIQkCKRAD4XLahJbU7crfXRwPtjAbIb527Q3xYqbUHZAVuw9kSZOmf/U+0x0ggpn1LPdn/SHs8qISXv4qyol5X'
    'lJsU9bpiv1nRqCuiJkWjrjhsVjQLRXlf1Gah2N8ftVVXbIzaqisKop7VFRujntUVBVHPC0W0L+p5oTjcH/WirtgY9aKuKIh6'
    'WVdsjHpZV8zqV0/YDwLlEXyMJGUEbSSRAqR8kZTzI2CXBfUY1D0uvylf/mWhPimdpFw+zq93BUaor9xjHunoE+4upw7tikM2'
    '3aqMykm5POLv34rHkHp8nl+xDcPDfNicVoZpjKcytEb3/gfZaTvRWhIAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

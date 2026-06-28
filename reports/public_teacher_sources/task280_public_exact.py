"""Task 280 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task280.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    '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'
    'tHsHYJ4oMMcFZlnBRGFvcNhJGB2z9dqhTO7JTsyc0sziwdMsDJhnBsyB8IeXic5MmOLM7MWXoJMriGAcTRpLfjX1IpHl1Z56'
    'KTLv9Y8qPsBuDkb96TSc9OJwEg4SAOSTw5xO44DNeN5Pxn2tUsmyKnt9dpKwTE56bK5/MZ7GJ8ctdu8J2aqfjGfT5nsH49Hp'
    'XjzeS+bx3ny2l0R70WgvGewNTj/8/cFsdPpTtWa/lfTjV53HbVbd43l/kLS2tmBf3VW6K5VK6+9Vq2bBVnU/A717Vqmcf1p5'
    'o8+b2Jfbtv5ZtdYYqBoDRWrb/bma9yriz3/JyH4huk9N2TIb4/MZ+f5KvhfatP5tW9vWLi+weZa7/7LfvML/y89l7svcl7kv'
    'c1/mvsx9mfsy9/9f7pZjVbfq+6R7pmvdRZ0tdCuvmGwFZdtMkjZJdK0qSjfZQ3P6byV7ZH7Semxt88do/Eure58ZPWFPffuV'
    'zytfVP5YeVb58vzLylfnX1W6593K1+dfV55/9vz8+a/PW/fZQ219X3UadXcwByKoYc6WsCSdSt0dtKlmjhgVO5m6Oxjlncyx'
    'dU2MWf6TSga4a63wgcv/DrpbuQS/tapsyI19+p7X3a4WfFqutcpC6Y6L7ntlJ0dFN12Ol7j8J/0YLnyfP+9SzU4F4nJc7ILR'
    'lcs9a0XUy9wU6W7lptn91DCzXdLdqmVP6w5/DzB7LrqrXPP9nfT10r4O21bV3gIWkn2BfXf59+A9SF8Hyyxe3klb1DIG/Gvz'
    '78t7mb6yEsMVw1C8mHPDeoGhY/aZ2QAWC7jKdHdfXgfdcablKy+vqXYwIa6n4hukt8lQ3DTauQzVLdKUZW/CFaawuIJjRKVs'
    '08oq3zUbr0z1KoIRbVNGxh3aHVWkkT1QhsYhOxJZHA7ZfCjQYU9LDuBupkEpq9+h7UhG+XdoG4fQNFLNu2YjUUFNVOOQEfEG'
    '6SnJB9Q9P4UBsccnM0OwQYaGu0VbdbLBrun2HBrqHezFoYGu6Y4aKr5BO2ioYht7Vwzpdd2dYsh3M70nHGmDnNY72T/pswa3'
    'jT6SrHY386d9ibfsDCma97RXpGDKYXtITnfbaBgp0JJtmzJtVKh1dPdHma7sAsFOiTJdod9N3aiRnUbvYIMGnURvy81fepId'
    '3ZhQnNpbktorT+0Vpfbyqf0lqf0lqf3y1H5Raj+fOliSOliSOihPHRSlDrLXLG6nF4r9YrEZ5Kax0y1UkFV55Sq/XBXkVHcK'
    'Nr0Ngxtkd9tQ3KK7qrxeYJZZbfgW6NTuarGf3NY1ddv89KTbzzmVozftMivNNruTpPtwOY1D9pALkqWbx0XJcIOlKJncDMlp'
    'bhsbt9lx3zZ2aQuqgpuzRWgWZbrrcl80h+W63AnNyW/gDmp2tb6Bm6ZZxU21M5gLdlPt9+VUt8guHlFWs0o3p7ypduPKVW5R'
    'SL3lt0RZmA8370pVeS9Hb/4t0RX6LZb4Lcr87mQ20pYaqKWvyEDtri0zWBLBvQiDexEG9yIMbjmGd82dNq1eQzXdUsuqb5F9'
    'MKGsmkrlm1Xur0Jl6+p/AQKvCC2/MwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

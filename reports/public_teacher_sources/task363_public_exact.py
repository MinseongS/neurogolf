"""Task 363 public-teacher exact source draft.

Generated from `public_candidates/lucifer_agi_circuit_20260628/extracted/task363.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAMD/QGoC/61WW2/bNhQWJdlSztLOYb2s7XKxg27r9LDZaa7dgKUqhgIBAgzJW18CRaZTJ47sSXKW5dfkz+xX7aE7vEmy'
    'LXndMDoKqXO+cyF5SH0uUOO5sWW8NLaN13+2oAu1QTSepOAk5xfDILwGh6mBGdxRRwzP+2hSOxsOQrZtwPfapM5NJgzqTPbc'
    'oM5H1fjLOPiD40Uv8Hw0jf8adFRaEwNU2m+DJPWWwExHT+GBmAh7mcNcOZgczCBNifwGVFZ8OtgvwMlsqCP6KtzPoP1QJx79'
    'Po5ZgsClU9abhOwkuPM+Azu4Y8mR9UAc73Nwrxkb9wY3yVNj3kE4GlY7MCsc7IMOTM24g7b1N/FlZjhIRKYVhiogNcN/Y/gV'
    'YCCwBskHasWdGC2dU5Z8CMZMKkOtDGeVh1C7Z/GoqztuDhxG69dJGsQpz+LtKAqDNMtCBd0ABYHadTIOImpfs6iHeOtNr4f6'
    'n7J1BPtiHPTAuscldYUMXznw16DnPQH7ZtRjW244itBdlD4QC61/gAyZxREBZGFes1hWQFaYW6CE1MK+vCp3geuEB0gY652H'
    'o0mU5rt7NrmZWl4izd6BLHVhTWtJOIqZCDCKbr0vYBmlERueizU9so9sXhcrYGPqyZGBP0vUGjr6FqQxFKJT5zYYDnrnF3wy'
    'v/w2CYbylCkprYlB+Xw8NZ9+go844mD1E3SJ0v5wMC6uzwlIT6CV1BkHgygVx/uTpoLT4NORU8EMlTnY9/3unvbGp+G8i1mQ'
    'srgIw4mIQfmhXdObhxNKO9SMOsXVWAcUoLBbbvwjqruo3i4c00HE0xbH1DgiR+b8aSclkbmXV7ORX6FwpzryDqp3/0vk5/Kw'
    '2f3BLUMfe7Nx91C4Xx73mTii8kyb0cGs5QEKD8stv0T1Pj6H1B6PEr6g1slkqBS7IITU7BcUz1CxjanedGgNa1dcaLmqz6+M'
    'm65UFazWQYJl16U27/K7YQNkNYCQ07p44ftnnU0uUN/K7w6lo/ZoIhF4ISKiDdl3ZQrSyZ18V4BkIwr8ojsfD4OI5dD3IGyh'
    'oBSS7SlJcaw/QrQehCnuYMk9SXSRKAjU0WN2EYqXSbr4GqTk0nvs2g3ntW0Q0/LxhHvLDfDFkTs2DcN7orTYfM0StJAQAF8z'
    'Bo8qJFr5ihxoGbEQqIhChqtJHJ9nhqtLnJAtuwRlhPi8DDFL8Wb48kPiregUbNuX3wev6boocg3RLMsXXwUPGqbPF+SYGNlU'
    'uRYvNa8l3lcJ+eujaoURIpjX1oiPmTwf+fwyRAjBn41ADIS3y3HDIAb+iX+izUK6xw2CzSg075GcX80X59WbCAtwRfZ4No57'
    'xic0MjMkparqNhu2OxN2Nuv/qXmrhZ2zm01fFfKp8X5T0Ue6Ck2X0AaYLsEH8Nngz0ULVKELhDmPuGrnVHHaiYbBVSvjiAsQ'
    'ih1WITbVR1wAoASwVbgkpnPNMe2cGy6A6KuhCrKeE0QKDYQsFyFcrWlgmbrJyR59DMuuQ12t4dJwXroivjEUwEWxrUXhjKip'
    'GVZBal9RybemZFs5KyuZ3SP+8K1QRGweIb2sSyI1vxFS/WKKHs2j5NQ2FZeq3M92zqE4ZKm8JgRkkQ/NlqqybWdEaJEXTYIW'
    'ZCIglTWzJjjQvLlMYk1QoKr15trtEq1cyDXBcRZ53lnoebfSc0OwGF4/S6rSuGRfSMyC5GAOcziFoYqXFGVrnHtURt5U9KNy'
    'OTc1MakCbChqUqVvZaxjgQfOIP5BX53iiyLfWJSHZBcziLpG+DYYjZW/AX2RUjVaEAAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

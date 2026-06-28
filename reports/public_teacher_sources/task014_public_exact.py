"""Task 014 public-teacher exact source draft.

Generated from `public_candidates/lucifer_agi_circuit_20260628/extracted/task014.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAL3/QGoC/41V227bRhDlkpRETZxE3LapEtuK4wRFwKKAJaOoYaCX0CjaCg3SKg8G+iJQy3VEV6JUkoKEPvVD+uBP7ewu'
    'KZEyycaGYGvOOTOzs5djAdWeaafaa22gXf7bha+hEYTLVQKmtxmc0yZbrMIkRkZ7xP0V4+9Xc+cxWH9yvvSDedwld0QfaPAS'
    'UiY0FiEf31DLm8Q8TMYTlJq/8jhG0glso9Bg07PxhDYmni85+rsIGV+ACkBjycPxTZaUGp5/i6TG9ZRHHHkDEBFKXAw230Qf'
    '3gah80C0HKiOylo8BuJSw11diI6uvDhx2qAni66u4KcIgxHFfaq7faS0RjyeektR7ctsJojQxnLmhVzU/clLsJ1CXblIxaD6'
    'crJXqa0Y32aMVrRYj6debrpvvY3Kx+MfjDvSKlvHTs4Ws2q5XiH/BrKyVI/OsvllQlyHXjE/FKYFqc7KhEaF8Pd8xf7HVnS6'
    'YMd8xlkynuEIx0Ho800u5a6X/sf2UpuyBzgOakRnwd6uNbc4Q5zV4FEf9f0aPeKsCu+C0IJogBrTuViU8X41QeQQxHd1sQJq'
    '/jzjoQDf+L6SMSFjQrbek61zsuuC7EgWUucd/4uLB/5I5ktRto8+AzPwN5HIEFMzirz1Lq/CmNAjxgpYDyQZt24WLOfeBrXI'
    'FTheX4WzIs4K+GvAKwVSRI14GpXcwfQufwUCB6mnrXga3CTcL6Ebin4omx6BHC2OIwhzj5YCr0AOEKdRAJ+CYIOI4mrwYMnV'
    'hmK1DmSFQTw60PqbR4vx6oI2gzAOfJ5/zhyQYkghaC5WiaDiJLw47l/kud9BFgVz6fnxHvlc3EzjN893PgFzvvD5qcUWYZx4'
    'YXJHDPlS49OLK4KMT0UCfN1EkR//WnmzgUbJB+eRpXdal7puuNIJnM8ton47xFVv/NDUtH++d44xCBJou+pVHwLRsp+iTr7q'
    'qHuV/OIcYKh1SYgrjpnz6ZbWdEhaIcDcNn63NaIbZqPZstrw4ODho8cdV55Ap2dRhOk92JY4czoq3bGbHSnnpSxiy3hF2pHz'
    'Ku2lLvmV82Tbse5mmzskmvNZLp7uzZAALtDC5VpqKg8PXLl5ueHprtqXIewqjrQ/nqfeQ58Ajoh2QLcIfgA/PfGZnEC6f5LR'
    'vs+4Pdna6P0c4i+5Pd0Zc0kWxXmeWnMl4Vh5clWNQ2G/AmxtQZLX4h2RsF4Cd6TzAlgoNrNulAkWy5Htoo/EY7HX6w59sbOk'
    '8gREUDKLuU8hWQ20jKo1HUnDqEHRLuq0/bppCZcQcLMcZvUwGk2tuh5GM6qEe+kjWiNf18uv6+S2Mh1xEprpSbCV1+RDVBlN'
    'LmaLGCvGqOQJh9jnFWNUNI12UnKY7OwwpS99BUVmQJ+ogNXMS2Ei1T1lD5XZTzLjKLlAivFiaxr/Szk/K6HIW+OaoHXs/wB9'
    'ijY2MQwAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

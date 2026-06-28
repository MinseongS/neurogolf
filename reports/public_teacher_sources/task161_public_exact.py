"""Task 161 public-teacher exact source draft.

Generated from `public_candidates/lucifer_agi_circuit_20260628/extracted/task161.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAPn+QGoC/41UTW/TQBDdtR3HngJNlq+oqtIAF2QJkaQJh14KqVClCC7pAYkLmGQpae0G7Lj0Z/AT+Cn8KO4ws7tO6iSm'
    'HJx43rw3uzuefR4ItsMes6esyw5+1+E5VKYXX7M5uONZ1P6Qmn8JlSS86uwLh0IUVE6i6Vh22Q2CrnCS2feCoL8QIKHdxZJp'
    'GEXI8Edyko3lSRYH2+CdS/l1Mo3TBvvJLZQ9AcUDO+2k9CPBRrng6fXaHVD7U2tjZWvcvrEuSmiHC0lys2QHOG7h835f2Kf7'
    'faRXjxMZzmWCuYc618Nc1KOc80amKSYaQGQgVNjTtIcp+9XFRElwn4ALC+dCholKTCjRAgVguU5fuPQ6Oy+utgNUCkxOWPH5'
    'smwTMBR2fD6lXRyF6TzwwZrPGq4+RQ8oh4TxFyS4r5LTt+FVsAVOeDVNGxxJm87+zHSYZKKKnYvDlBZ1j8P5F5kU9Jquuqvp'
    'qPw3/UU+G2baXBSnWVz+RTbquoJm8H90j8CsgD1u90UFg9UWPwJTzFAwWKfkjdAcOnK8RjGHNxSKipSWbpKDP9nKF7PyuVNJ'
    'sLIO7uMyjLLlrDxRjY5B40hpi8ooml5Iuh7vsNVSk2jdAulonaTbAFoPTtbt94U1ulwhqUaA1uekowIJB3t0CQjScSe0iI0j'
    'prqhAHAjeSmjVLizbI5fj7Svv2Vh1GWCnwZ1z65VD2xm2QNjFcG2ZyFkIaKvaw5wDXSWgKWBblAzABsYg8oR4AaRwS2PI8L5'
    'gMwlj4CiRY4NyG6C2zUk4c0fOox9fGnCngp/5GFHhb8OTdhWITsMtrCSNcCPN+Q88NVre8hZsIWvqn9D/idoetwDfIhq2jME'
    'xi3bqbhVzx+x93tm1sUDuOdxUQPL4/gAPk16PrXA9FMx/HXGWVNf4pUKvslzytOt3ZDnSi+0FwsAD/MO1T7bRuO7Bvhn98jW'
    'xB3ABgovL0Foso7WlTkquW/kdW2VKxC6XQF6oC1yrWBj4YmU8YsbiNfR+9oMCXaL+6I7SStW1Yr8bHnZV9rDF+1bXvYNFN3B'
    'Vm48pUVaue+U1tgzF3XlMy9L7JlLuoHA8zmgs5QW0HMSl+qFNiTVH8v0565xlwK4Z8xEVbI2b/WohKCX2iUvKZXvKpcp0za1'
    '3ZSomwMHWK3+F5erQakJCQAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

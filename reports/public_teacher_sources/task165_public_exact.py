"""Task 165 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task165.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIADzpQGoC/+1W3W7bNhS2bDmSj3/icO4QFG3mCGtWaFuXNOi2DAXquO0aCN2wNgMG7MZjZMaRo0iuJKfFrvYgu9jVHmdP'
    'socYSZGyRMkbMOSytmXyHH48P/w5+kzzmz/vwh40vWCxTKAVJzhK4snZDAwSTFkH1c9mVvPU91wCu0AFaJ9jPyYTN/TDCBlB'
    'GPxKotBqPn+zxD48BqmBrjAWkZkXBtDmBlMB9QRIyNLBV6AMoEFRnpz7IU4s/SmOE7sF9STcrv+h1eF3DSqRAG8Ovnw0iV3s'
    'E+jwPscsv4Zbl15CJgm5WviYdvgYVa+fsHYEIW5K+nfDZZBY7VcvvYDg6GkYXMMhVECEQfeImTD5+IWXWMaLiNCAIvgUMiXq'
    'yV7FAgBbgFNQILBJfG/mnflkckmigPiolylSG10W2o8RDuJFGBN7C/QFnsaj2qg+ghFdVYNGoMxBppQLEbRYBDurnc9QSI8J'
    'mVqN42AKQ+ACMtg/Tbm8ic9AjqE+dhPvmh+y5VXA0K3XZLp0yXf4nd0GHb8jMY/R3qSrRMhi6l3F2xqz8hBKk1G3oCnHvgdF'
    'RC6DjXPP9x/upzlYIERos7WaMOFwP8UcUswPeArjVRY93onCt/TUhBGpyqGh5lBj8YxAmYpMKVsbx9Ess+DFfOnKFo6yKCCb'
    'ijYzo9fYX5LY2niBkwsSFWzBc1BxqMMVdG0mES5HoFdG8AAKs0QGVLKMb+mNS0iQGeD7ZoEp7oafC9lgUbBJjePplBYgI3mb'
    'h9AeMpiHDPIyZ6btXuCAHn0uSEsg8SIrL5jS0kPXgl4HFyeFrGgSojIWsKgbL3AUkyOxr2a6jN8/g+NCdYQijp4HLh7sixuo'
    'utRSl+I4gQIXoaCNcJnQ1mr+RH0SZCQ4vqSFxP67YWr0e8ts92Gs3n7nr0btJj+PKzTyW6XL699rb0BrH5h63xiv3tjOcN1e'
    '1UVrf8GnyDe7M9TEgGwHokVywl1+oui3r43zB9vR+fAjbq/4ni+HoSmyfcin5flAOZSeaPty0qlpskm5suuM/ith9QOi7Uij'
    'v9DcWiJDY5wVDufkpq6JPSl4kNXLOakrof7f1nYLDvIVzznRFLAsAbpom6LdEK0hWlO0LelkQPc/R3/Y9v/2xP6gXx8XiJCj'
    '1ewtqswRG0dr2U94eLpp0KFq1uXcybZe0zT+z3+p5uePZOn7EAamhvpQNzX6AH122HM2BFEU1yHmdxh1VUbZM6BPf76bcRcO'
    'aVVA7peoaRnZ5cgH1VSU4+sV+M+qyGEFmiUDcyvHCssRpJj7KhnkSKhGKgyvjEzzt3KcaN0a7QiKt258d0WNyumlELuCvZWx'
    'Gsd+orC2Cr8pcChp29rIhvKlW4FID9DHJUqGoE/D6uRtzW/nqEsPOqaBTDk+/7zMq27DNjUxUAJOg95T+BMLzCglp2UuGaNR'
    'Xd7LWA93ZWSuWgUT91ac6N9gtsKC1mN1tjdF7lO8ehmYncAizam4whw51qHW3/oH/14GnLEOAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

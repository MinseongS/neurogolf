"""Task 175 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task175.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAKL8QGoC/+2Yz2/TSBTHxz9iO49AXXeFItMtrHeFwBdK0QKqQCphoSgSHOhtpZVlx1PqNolD7Oxm98SfwJ/Ahf+C/4Mz'
    '/wG3ve6MPXbtiROc0O6prV5mPPN97/NmPLbkp4Ghxm50cvfBr7tfbNiFRjAcTWKA0Y7rRLE7jiPQaB8P/YiMjkMPO+4UR4ZE'
    'Rk36YzUO+kEPl3y9gq9X6etRXy/zfQw0kqFSUuBPzaxjKU/Gb166U/sSyO40iNrCB0G010A7wXjkB4N0IHX3qLuXuXtLuF+H'
    'jAeZpyG62yYxSyLu8JotzbjcC/vh2BmNcYSHsVm+tJqvsT/pYQpco0Ac7aE9cU/6IKglKKLQP6DsbTROnIE7NdNmJnPEZ54M'
    'tGE9wn3ci52+G8VOMPTxNF3TLSDZG4q77QT3dswmaeOQdi35KVHaTRDjsC1S5S4wlUFVUc/tu2PztGupB28nGP+D7fV8TQJb'
    'FdyBU2ECO7x7P4ORbgkGFHYb0vXR1VKtdjJX2oErnhthZxAMJ5ET/xUCA0Dqa1wehVEQB39ih+rM8qUlHUwG8BzKo7lruvXe'
    '304v9LFZviT3PPTpxh8OQj/dzX0oSwy9dOkED82N8gjd7YelRUk00CuY8YSN3pE7HOI+OXeREx0FhzH2jSvhEB+FcZ4id201'
    'nr2duH14AtwEQLJpyUkwlHASk2NrstZS9t34CI/zQ0Xvf/782x83tS1tS1c6hRDd95sCSv8EZiIziZnMrMFMYaYy05g1mQGx'
    'SzUsYyKOX8yhmEcxl2I+xZzqcIUKNs/nc+Dz4HNZhlvFruJX5cDnUYcr1mDP48/LYRXuIvYivrAkVyr41WV/i1+XKy64t2gF'
    'fh2uXOCuykbnwF2GjZbgNri9Pgv2WXKXYdfhKhXc72XX5cqFZ0+o+Qx9L1dl3GXW/C32eXIXsetwNe5snQX7/+BWsetwm4w7'
    'b69XYdflKhx3lTUX2XW4UIN7Hu/JC7uwC7uwZc3e0QTy39Khw32gd9voExE8Qnuog35Dz9BztI9evHvx9bN9m3iAJuhSp+pr'
    'twv/IkGU5IaiaratSbraKZSLuu3s9SeyVmJtrs2LVadaxPnYtxJtXszqtoF7Wc9G9Waiilz006geF1Xiov9+PSsiXYUfNMHQ'
    'QdQEYkBsi5p3A9h3eqJoziqOf0zLZOUATdYK6bQ3d/qnvMyVSNRcIpQl3kLJZlJYmjd7ja9qAWgkGZku43gtK/4oIBNvdHwj'
    'Lz3ReGJFvI1SgYm4icRNzypByQiQkbWssJMN/MwVfgwDdDLRKgRvUVG5qlMlujlbsUl0Eqf7ha/EJKpmrqJ3sdWRAemt/wA2'
    '+tx+dRUAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

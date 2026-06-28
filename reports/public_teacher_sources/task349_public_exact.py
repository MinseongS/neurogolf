"""Task 349 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task349.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJfnQGoC/8WWS2/TQBDHvfY6diZ9GJNWRYKAzAHwBfw4NOWA5YpLBBIPCSQ4WE6yKFZSO42dUnHig/TAV+LTIIRQ2XW8'
    'xi1OKqFItbXKZPbv386Md72rqgfnO/AI5CiezjOAwagbpFk4y1JQmU3iYapL1DLkt5NoQKBbSHXlJJxEw+CT0XxDhvMBeRme'
    'mi3A4SlJPfQNKeY2qGNCpsPoKN2jDhHuA3+GP9w38GGYZmYTxCzZazJRB9hoeoMNPt+/0C+y/u8Iij6AY8el0Q7CCYGt3D4N'
    'vpBZwvq2RsGYzGIyCaL9YPCkRvt5oY1WczZHQT8K0yBybIrRW6NgmqTUYrG1Xr+IYhLODpP4xNyBjWLAdBROiYe8DVaEG4Cn'
    '4TD1BOoQPGAuDZQ0m0VDVqe8Uv+Xk7WenKwyJ+v6c7LXk5Nd5mRff07OenJyypyc68/JXU9ObpmTu9acPkJ1nerKKJwk1DYU'
    '+o16lSSTf+CSh6tw5IkFaiXcqsCt5XDZU6twkTpET7wKblfg9nK4crEskqfSJl0FdypwZzm86WlVOPZatOGr4G4F7i6Ht7x2'
    'FS57W7TJ9fCnwN8hNyxu2NxwuOEWg9MZJdHB4R2f4LrSJ+ER8y8NqrN46zyoO3SO0bs+KAM4T2/kRs1uZgCPRW/kRo3mXhmf'
    'zH5rFI+hGAAaSUzy5XhxWdHtMpkkM8uQ34/IjND9vBgN1Gw0I/kjhaSQOlz6ABajghJHcUXoFMIuF9p89+4Xii60UhJnEase'
    'DaGVO4Ncw58xoOoFPM5jTeYZPUAY8vPjeTjRtSxMxyybXHkUTs1dFWvKARaaguBXDiRme+FH0On45eHEbGvIwILw9Zlf+e6Y'
    'O5poIsG/VCfqlv66yy+WucnUyC+qa26zv5Jf1s7cYo6mz0tk3qT/xXPkVwtg3laRCrQh2gkCEiUsNxS16edZmw9ZD703NMm/'
    'dDjpbZwhdCYUl3mLEhr+xYNHD5/TawXEyiElpg5i9fDv1RC7gBSYOojdw79WQ5wSkmPqIE4P/1wNcSsQiqmDuD38g0I+3OUH'
    '0l1oq0jXQFQRbUBbh7U+XV+LGZcrmv8qfAyCtvkHopYAmAoLAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

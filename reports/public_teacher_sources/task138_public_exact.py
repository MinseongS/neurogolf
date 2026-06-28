"""Task 138 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task138.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAETpQGoC/9VWa27bRhAWn6LGTiOvHVuIYydlE9RlU0SM+6MI0FZWWwRgEiSoUxQICgj0ck0TlkmZpGyjv3IUX6E36Bl6'
    'gh6lM8uHHrZS518r+9NjXt/s7OwOLXj2Vwd+BCOKR+Mc9EDwF8zg4eHuU1v/IYnPnDuwfCzSWAwH2ZE/Ej2lp1wqTWcF9JEf'
    'ZL1G8YciuAeFI1N5iM5+ljstUPOko14qKnQBxdLJBWP8zcDtMo2HI1t74wfOKugnSSBsiydxlvtxfqlosFNlpabHTEszXmbU'
    'hmaWp1EgMkxmk5gnlhwt+XWWmzJtzIICYbRuZJt7afjKv3CWQPcvokym6dwG61iIURCdZJ0G5e1VHu4NPZwOrGRiKHg+GGIN'
    'BlEciIuOUtSAksMMr2HXFrEXHu4NPT7Avgm0bFo7n9kek5QdUnLQs9Ouy9S0azf3T8dC/C6km0tu7gI3d+LmzrhxYuML2PiE'
    'jc+ycWLjC9j4hI1PsVEecYid0mUWbvjFIPXPbW0vCOAzqAWodpmZnvnDKLCXXoose53+dDr2h/A5lOIpYzPafTrY7TKdJLbx'
    '65FIJzwcefg8D695OPLw63l4ycOv8PApnh2gwwGSG0OlyWiQ2uZzP0ftTPvBF1CqQQbAMPjriin1CqxRAWSNjDQb+bGt7Y8P'
    'SMpduSKDT6R3i4UalBye1DDHPXqeCj8XKWyUOhkEG2OY2zqtslbwQsFrxRpQBCBTtI+QYy8OaiknKa+kW0AWeEfEXXeXqh9/'
    'bbd+ibO6QaQIyIG1ojgW6YmfHRe+OzCRgKwE6HjX4GZJMU+GVYG/hVrETJ/n0ZmwWz+LYMwFnbFbVDe8NtSeRrfd9CmTZ+lx'
    'Gb1J9cmTETPxLU3Or9+jx1Cqi3q60pqSuXabvixjl7bNDC+whaGfQKUvGqkbsuZQHOYLo085lPuELR8eLXbYgrI+UGbNdPw8'
    'sY2ipe/X6oqXGfSlNvi0NqiJ8BDSt9rkLhQuUMqZkR8O/dBWX6d44iRdaYGdW2ueTG2hkQkRvLVbb1M/zkZJJuSAwkbA4aT0'
    'tJ5Kl/8jKOJCYT3lrpOgao3vQP5k2uF5YDexG94kyfDKHNyYnYPrkzlY+6fi7Mb+5L1e+D+AYpFAAYCyYOZhNBz69e3wFZQC'
    'ZtDnv637Yb1uaT3rfFAFfTh9dgpV0Yg0p5PKapcupgQMTHr/XTXIdR7udj8wybdAWtBN5cehYGYyznFcl9vPlNB5ZikWIJS2'
    '0pcPId5OQ77ef49vPfxHvEdcIv5E/I1o7DUa7T3npWW1m335ZOH1Gh/5UuY+naW22pd3hqc0nGX8UazRU8D5Qymz1DFLfCjx'
    'LpWrASnh/w6qnHVZWXw8+j/kfM9SaUNpynvt+Y1yQmujbfZp0Hi/VQoVoSF0hIEwEU2EhWghALGEWEbcQnyCuI2g6CsIhlhF'
    'rCHuINaJiGHNzH45nj19m2QrUlZMRU+X+TyyNMy2GFdeZz5drUp7VXpW86L0nQrneroyx9oNS7NXssOLM/fxLd6c+3S268Om'
    '9ssj6UFDUTXdMJtW69398nGarcOapbA2qJaCAMQ24eABlCdYWrSuWvR1aLRX/gHMuRp8XQwAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

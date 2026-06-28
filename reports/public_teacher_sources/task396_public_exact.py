"""Task 396 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task396.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIADzpQGoC/51V3VLbVhA+K5tYbEIxBxconZKWzrSM2pl2Mp2mZTJTW/w7EAgmgeALRljCGP/ISDYYrnzZiz5EHiWP0EfI'
    'I/Syl/2OZGGZYNoGzRqd3e/bb3clnaPrUsyKebEgnojFPzM8zSOVRrPdYrIk5RBJLrmNiyeCZ5lyTMdMJSZbkonQSKFWKTmI'
    'fcFkSloK0JbfMh6y1nJn+Ey8JQ3h75mWJC0jnNqyOjuuWzOm+VHV8RpO7cg/tZpOlrJJhU6FQstMDtMJU1nSSlxoEpmYTiWt'
    'xirLMK0yVSStKejKeduqhWmWgnrPgnrX42nAWGFal7QRZ0imDaY1SXl4E7mGDd80Uz4AI/1zBd4/dTyV4lem55I24Rrddex2'
    'yUFnxjgnrY7jZ7VsIujGmGC96jhNu1L3ZygaRyaobFPSVlwdM9yS9GLYDBeYXjBVJW1HnRuTnGxatp8VwZWK5hcia5J27kSm'
    'siJCoodtSS8/pgdQdyTtfgwVjb5kqksqgP3JmudYLcfb9mJzKEjaGzaH75j2mBqSXvW1C+36UKndQOr1MKnXkvbvkdoPpA7+'
    'g9TTYJZvBgcyFg3kw3GIiPgT0xtJhyA+yHnlG1bFnwlY2t2sOaZDqVnWsNp/Uc9Hs47vridxTz0/M2iglv5nRY/BK4FnDyvp'
    'x5udxQbMGT7TgZQOsCfDUmI72GRyASmrT7bQPoZziqkJYhne08ENAQ58yPh3glgl/jVn4KwwncN/pjJttWu9UZzBVb1jFIl/'
    'GUUVvNqwupVcLSy8Huw1th3uCx4CFryNuBdLJh9et1+a8rphhma/9SADvYLzXElvOr7fa/wc1mRqIeQNNh5oqifiD2r6oWZr'
    'ULMVarZvax7AeXFL8wLWDjUv45oz8IPShr+jzoRdJzgDepFLpgtErgYj3yDSYbpE5Fo9DWxrJat162kEOOoAegVc7j7cpwBd'
    'w3IAmv3GHbU1wwOvOs14zavYBaverDnGFI9ZtUq5cVRycW550advZDhZd21nPtVwLM/xW8qbMGb5EXZcu9IoHwXRkWvHc/0g'
    'FryiyK+OAM1ajr+in8O/DKurOIIr8bEpEg4i1dpqnKSGhvPvWj5w2y18X6qbHQvdSCobT3XSGUZpMsnKLwjR/U0InAMiC+vC'
    '3sLewd7DRE6INOzLnPFQ19KpRU0Ik46jhZQmlaKFljDJNpAbCwLKie7HTTqJ7jUTNXyr9PWknkyzMdP9I9HthBf0+j8mnRo6'
    'ACpVJdIYR64z43HAJx1hk6r5tDgQe2JHbIp1sYxOnvUAOiQBqA0B6MEUAKgHgENR7BbfFd8X/yr+XYzKxYwaxqgCuXmtx2O1'
    'bOYzqPEZkplIuSJWxZpY764bcyHAmBRRU/3LpHPjs7BoD5K36KGMn9d+/yG8bUExa4xhSqnFJKFWk9o3S12VdmFM9xs1dNH7'
    'M+kyCPQajAc6xtc3zx8SV+jigx7EhjGBKlOLPd7UlEnXu+Lwq952Lac4o5NMs6YTjGFzymbF8Tz3XrkAM3oXxkyySE/8A5KC'
    'RG1lCgAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

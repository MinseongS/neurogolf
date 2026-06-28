"""Task 213 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task213.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEfpQGoC/5VYa27bRhDWw7KoseMYbOIHmzqO6hSB0ABaGo3bNI1jtUkAoW2KpkWB/hFYia6UyKQiUmOjv3KUHKVH6QF6'
    'h3b5WHIf5Eo2THF3dmY4/HZnvl0aYFYe/3sCfWhMvNkihM2hv/DCwaU7+XMcms3xIO5brNFefz7xgsVFZx8M993CCSe+1wZv'
    'OL78fPjwqTf+UK3DC87X1J8zX43xgBpayW0VP2UxIYsJV4rpUhcTJjHhCjHFfh4AQyJ6n5kfWMmtvfatE4SdFtRCf6/1oVqL'
    'NJFpYqKJxZoPIcEkdw3jwWQ0OJ/6Tmhx7Xb9uwlG6pioM/+AnDoq6pwHcz1qH3et9C4EU0uC4TyY65iqY4n6KaSeYD0InXnY'
    'hTXXG3Wh4VxNAgKNIHRnx6YR6QSDWdfKWu3G6+lk6MJTSABcZk9VUvukxeyPIRNFsEWt80iRawtRQxT198ANp88yb6boU9Hg'
    'fEYeRS6YoN362R0thu5rujpugvHWdWejyUWwV428/QCyKeyPB+hMKYxUMKQPP/enI6riBfHcsiGLa7ebL+euE7pzigiDNAUh'
    'QoSUIEoyRAlD5BuGqN48xoxkgBIVUMIBSjhAiR5QogJKZEDJyoCSawBKOEBJDqi8Ru0YErsEUTtD1C5Zo6X2MWp2BqmtQmpz'
    'kNocpLYeUluF1JYhtVeG1L4GpDYHqZ1D2s2zTl0u/rybLZeo3a69mkMPOAmHh5m0AnfqDkN3NJg7l1aBLPbxNXAJw7WJeYO1'
    'HVo8iCV22/Uzb1RqbIvGtmhsq8ak1JiIxiQ17oEYj9i1za3MnTOJoJP68aufgiQF8UnmJj9sCb3YwRcgyKKVlPQ8PzGRBe36'
    'j35IOViWQ8HsRK/Ayyypn+DwWJiBjAlgw1+EwWTkDsgVSQgqSAkqoGX+t7E7d+FEmICs5kGqFm1UojuxWIMZvuBnKzO0gelF'
    'oUciPnS+z/z0QXonkBTF12hRRnffRQpW3mS+ugqo5obnh4NUaPGdZBpoCcMVaBYzmkWVZnE5zWJGs6jSLGY0ixzNop5mUaVZ'
    'lGkWV6ZZvAbNIkezWEyzuJxmMaNZVGkWl9IsZjSLKs1iVjeRo1nU0yyqNIsyzeLKNIvXoFnkaBaLaRZXoFnMaBZVmsXlNIsZ'
    'zaJKs5jRLHI0i3qaRZVmUaZZXJlm8Ro0ixzNYjHNYkaz8nJJaBYVmkWOZjM8TCygWSym2RfAFyAoUDO3UKr3WFzvkav3WFzv'
    'Ma33KNd75Oo9ZvUe03qPrN6jUu+Rq/eY1Xtk9R6leo9l9R6leo+6eo95vUe53j+CnANg3blyA3IcbZYi0dy/DCyu3W796gXv'
    'Fq77l0t3HXIMmXEiT4zzNm98CpxX2BqOHc9zpxE0CzeItgvJQTh2IfTajef08Dul6Zi/iWqPqQX9pfZ8j9l/CYJb4OI0N+jv'
    '4NwZhv78xOI78QJ8onCt8ABzg/7m1lwntj4B3iHw4yYM/YsZbX91dWJx7WS9vgJOBJszh872kAzGl/YxtM6daZCuWDrvs0Vo'
    'pfd2/Sdn1PkI1i78kds24sx2vPBDtW42Qyd4a5PjzhMDtqs94TNG/0El/nt/Kl6qLLfOP1zw1pVn9J9e758lsr/p/Z+ofVap'
    'bJ917htVo0Wv6natJ81iv1Wt1dca602j1bkVq7R6+Yv2q5XOTmrIr/V+9b/ObSpt9pJq2TeqyYMrnQOjRsXpIu1vM3mdjadm'
    'cSXvG5l4JxanO5O+USmS232jxuS3Ynm8g+kbt1UpDWlHlVIPu0z6i2FQqTDF/WfsuSzsZX+70r3zKX0W9MoLfr9WOfv9bvol'
    'ytwBGpu5DTWjSi+g10F0/XEI6cqKNVqqxpv9/EvRFmxSJ0aqcvBmN/2WpAzs55+LCmyw0GY3PffGAy3ZomjgjvCxSfZ3R/i2'
    'JI/usWN6PFITR7B4xMqPEdJYMxlLt6xinM0kTrYzjUeBG72nfNCRVKqJAzbNkvsqFxbRhEW0YRFtWGR5WEQTlq0Jy9aGZWvD'
    'speHZSth3eE/DSgPPyo+cEpad6UztvKQu8qpW6tAVIVD+QiuaBxIZ2x5/J567it8CP+2ypvuZWdeeQr3uUOtNHQon1YVjY+5'
    'PYoy+ImwIyxaV6jJQtRkIWqzEJdnIWqzEDVZiJosRG0W4vIsRG0WoiYLUZOFqM1CXJ6FqM1C1GYhLs/CQ3nDWrR6sXT1Yvnq'
    'xaWrF0tX7xG/E45JtZaRaqwRv/+RsEEVqTfX+kzc1Ep6LV5P2LKqekls94XNaqm7++I2tszbEb+DLXFGN0NQ2b7xPy5oqnn5'
    'GwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

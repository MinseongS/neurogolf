"""Task 054 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task054.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJX8QGoC/+Wd73JbN3bARUmWqKN/9JUlO9wkm6r9sMP2gw/g7LrZ7cRWnLpmEu9uvJvNOuOwlHQtcSKRCknRbj+l05l2'
    'pp0+Q/MQfYC+SGf6CPsE3QK4wL0AzgFJyfm23lUucQAcAOcenIvfxb1kHbLVcXf0zd33733wX/9Tg7+DG73+xeUYNs66h/lZ'
    '52jQn3ReZWtF6qUUzerj/vJHKre1Cxvf5MO+ko1Ouxf5g9qD2ve1VXgfqpJZvfh4eb+5afV2R2OVVCrUh9YaLI4Hdxa/ry3C'
    'AZRlVVeOX3fuwqrR20FYHb8adHo/vZfB0UA1OOwMB6+a3uf9G8/Oekc5PARPSLRA93Vv1HnVOx6fZjcOT3Sn1mzxwxOn4q/8'
    'bphC2Zo6nCtTdQ6bq/bj/o2Pv73snsFfQ5UJq/+YDwe63sqgn5uK/cG4YxsqP+7f+N1pPszh2Bl87bDbP+50X+ejbM9IlO3P'
    'BkP138v+eGRM//bpYBSLh4Pzjim+v/Z5fnx5lD+7PG9tQ/2bPL847p2P7tS0Vb+AhM7s1uHgdSjtqZZ2aUs9c8Krs7Wi9QrY'
    'KApNumeX+ciZCtShkB826+6zM9YJeNlQN9ZSyoHtSdYk0v5AW1D3co/Pc6Z9AVMqZ6s6TzlTs1EV6g5Pzruv91ceDk8+675u'
    'rcOy9hVjQmpTAU5FtqI/qNO75bWnLEW9Gz23spWydWNr61lrZcKZ6xfgF6i8q3KlrD4YqwHrDpSfnA0+g1IE68P85d3OaNwd'
    'jkewZhJ5/3jke96GkR4NBxfGW8uUmxY9CEpAQ6U6o/Pu2ZnT60m08o701WdFZX0qyka2I5lr6gkwpbOdWKb9YLPqE+ulXwJX'
    'ze/YlpevNUKVnjqvYvOib15kzYuBeZExL76JeZExLybNi4x5kTMvzjQvzjAvRubFa5hX+OYVrHlFYF7BmFdcwbz3YvMKxrwi'
    'aV7BmFdw5hUzzStmmFdE5hVzmPcxRH4P0Ymy8+Jk3DHyw+aGn95ffTzMu+N8CL+FqOBMxWbW+y6x7glc8GpDXAyioTo9RcNC'
    '9XAzEFRd/BTionGfzfntqKXBWFl42Ds5VeG12Yhl+0sP+8cgSeUifZaHlrLp/aWngzH8KtWFslZ2yzZ3OBiP1YX9LH+pO5FR'
    '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'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

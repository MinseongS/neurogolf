"""Task 218 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task218.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEXpQGoC/51WW2/cRBT2ZROPT5rFdaAJprTBIiryW1KppAjRzZKLMBQirhIvlhvPJt449tbjTaM89afklwAtVS9/gGd+'
    'CjOzvs1488JGo5zbfHPON+M5g+CLfz6EI1iI08m0gPeOTsI0xUnwDMfHJwWxl0h4NklwcJzHkWOVylGWZHkQR8TtfZ2l554N'
    'ZhQnYRFnKRmYA/NKNTwLDFLQSZgM9IFOLfAVtMHsfksJptuOpFPokBSeCVqRrWlXqgbfCfPByrNnQRSPRnWuqLI4tVQmeBN6'
    'kzAiA3WgsD+WjYxGa5LQKotTSxKaMsNjaI+gXtLmiTEej3EUjKZJ4nQsQnEmK24POkFgMwspwrwIHgThBSab922ztjmN6Bo/'
    'PZ1ifIlhBxorLOVhehqQoyzHBNAlzrNgtPnABh7BrU5Ldhd+O8E5hhhaRlgssslpcGr3mY3KwXmYTDGxDabH0cWMahbk9n7O'
    'Jt96S9ALL2KyRmnRvD4YSZgfY1KsqUxfhkWS5QWOuAqfgwzLs6cyPWyN2GWL0l3tic13TqRbtnQB9qETBCvMMqN7c6vmuzY6'
    'jSjwXVuv5ZtHlHw3covvxtjwzWxtvpnO+S4d/5NvCZZnX/Jdi126vgfp64Rq/+sLgurEuVkqk7A4OuEmd/EgLGiddZ78U/4B'
    '2tOgKs5ebl8wxFkR4GbGDqDOAB9Dc1xgtRbZnGBzu9rMZcHhiKpr/pKSclsfQ8MGrNYiyzbYfFjDCQ5HVNtwuyAuBWKovVR6'
    'cJIQx54p2bSgF/LM5uo7aQSH0I4DkSsw+HmbbtvLZyE5xVHNIVODOK0XZxyWR+8hiMFgTMIEFwW2l7IUn2RF8CSjPLUVd2Hv'
    '6TRM4EdoWwHRCzFglyIsjMKEYHtxlr9jMUeRBSme5tlxloxc/TCMvBXonWURdulpTunHkxZXqm73C5rMFt0ukiXndIf/UJGK'
    'AGlIs9Sh3Jn8K1Xp/J4/kgwDSZX055J+Jel/Sfq/kq7siKol6N6viFWgIYPm32lX/jZd/09FWX+hKG9fKsrlK2Vw+7Vy+Pcb'
    '5dPi3ayWdYr3dkh9u9S3T30H1PdNiWtwXjqNq8Ll81+UGC9LnFcl1usS702J+Y7j3kKqZQzLK8hHelXHJ7QCGLavN7+vHCj7'
    'yp6yqwwpyV96Fg2orzxfo5P6ljasDqWvKt4dmrLJEmf28pz5pqrpvYVFA5neIUJ08fog+TXVc/Z57u8j6b93wzI9VRnOTqS3'
    'hXSKP6en+mvySnXh9/mceY2hO0mrJt3jLF53CflIlQKvuV58VK3w+93yeWbfgveRalugIZUOoOMOG0/WofzeeITZjRhviI8v'
    'EYgNg43xZ/JFzyO1OZFu68UzH01lMXWb7sbwuLHXffhIRTR4q63XjQ2AaFCPg6y13yzcA6XndueJ0Xj18QdNH2NmozSvtrpJ'
    'ax2dJSs/G+YkOytstfU0kJNtGr6crNSfpWSrPiklW58gIdkNocnO2UidL3pP6iTXBOrju1IXs/twgy6GygCVBYh9rQngKOOP'
    'hQ4mzddZJkI7ujaTDaH5SDtgVmHDHijW8n+toU/04QwAAA=='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

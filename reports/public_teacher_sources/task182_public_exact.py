"""Task 182 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task182.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAFzoQGoC/5VW227bRhAV76uxk8pr2ZaNxlbkoEDZC2KnAYwWbWQXQQCiaROnQYC+CDS5smnLpEtSqpsnf0Ie+gH+hf5B'
    'H/sDfe+ftLMXShQlFY0FYjhnLjucy44JfPnbBrwBK4qvhjnowS/UDqKw/2i/Y36bxCOXQj2MBn4eJXHWhS7cao67BssXLI3Z'
    'oJed+Vesq3d1Dq+AeeWHWbcmfwjBNihv1ECKLv0sd+ug50kLTXT4Yiy30yxIUtZZOhyx1D9lL5JkMHOQ021yryWr4H9YNbsO'
    't9oHdQbV06hjH6anz/1rdwlM/zrKRDjuB0AuGLsKo8usVePxoU2gbIJZG2OuzTqgfzCyYA8PSjvOMRNhcDwo8CCY4Fuon4Kd'
    '5WkUMuqk8qVjPB8OYBMKHo0DavVP/AxFh2EILZAcmEm/n1Ezj8JrKbkPPNkAfSybTAG1As5MztwEiYAwo3o+6tjP/PyMpfAh'
    'VmcE5tve8IA6+aWfXfROOs6zlPk5Sh9AgVEiX4YHs2X9CMZCWpdvkeioiZ7N9Q5gIgXDv35IIRYBCvX6MQuHAXs1vJxKssYt'
    'dycnlM4iF+KbMSSViDEA9oXMhMlbY6oo+LVGwGJqsDgYZ2EXSpGA8+vbK/6CCTlLRWzCfwcKntr8ZV4qtmQx7CRmPEIz2MN0'
    'Wk9/HvoDnA7BYnX25tpegpSAcZ0FsiQg4peA8ikZA0OklpqGl99FMfNTMcDVabC6VnlYdfnjA/IUpD2tC9LDue842O9zh2qx'
    'Gxcm9qDyQi1k8MPvqj76IZUZuAdSoBJRT1mQDJIUNY3DOMRumyDA6yOSSe1kmPfwOrHeYKkYfAUKAJvH8uihoPuPH1OH4/je'
    'MV74obsK5mWCk0UCvMtyP85vNQPaUCjhqJ+N/EEm3ONlqKqENcfu2jvYd5ca+pEogqfV3DvIqAJ4mub+rhGNANGJ3tCO8Bb1'
    'brXazN/NkwrQrbAV/qbC31b4Pyr83xW+djjNNqZ4d5VoDeeIX0geKaJ11wWobiOPNAv8L41sokBcNt6f2prC1xXdULSl6Kai'
    'Hxf2in6i6KeKfqbo14p+o+iTSoaKyF8qeqzoK0V/VPS1okzRvqKnip4pGhVxbYnvLd2VHnmnTScI7yWPjLP2OTF5guR14rWL'
    'xBXUqtDCCd4wHlkug/ZRca145g0H72Dr8Fn2TN4qLmCH8an2tBv3e0L4obK9vW7tPf+gQmX3yiHxtH/cbdG82MIcllPgQU3T'
    'DdOyHVL/aUf9f0DXoUk02gCdaPgAPtv8OWmDGhqhUZ/VOG+Pd/a0D/40+XN+Tw43F+tzxO3x+p51cJdTccQiDaF13uKrmVJo'
    'EIcul6VcEsyXNPh6pgAEJWaB4C4uI2vjNT0Fr6oVPQVStXMn2OZ5U61jugR1/HgLDPLO4OfkI6GnK721yfblcF3B66UVWFbf'
    'KK1XIbCVoFXebiWJxl0VK3PK1bZcPHOqY/HnfEVczyUTkZRiO5ZPaI53Qlm5WIXT7TOp/o5ahQvbY6dYXosUdktr6b+8iH20'
    'MI7d0kZaqNQuFtLCc+6Pt84cFTEwRybUGiv/AtGMqJ+fCwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

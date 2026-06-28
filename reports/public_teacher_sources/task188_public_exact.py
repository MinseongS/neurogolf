"""Task 188 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task188.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAEjpQGoC/52X3W7jRBTH48RJ7FPKBlOh1Uh8yAgJWUL112anBQSbihsE7AIXSAhhuYnB1qZx1nbaaK94lL4bL8KMz9gz'
    'iVNa9SIzx8fn/PyfOTP2xIDzfwl8DsNstd5UMJhHLm88a/RXVpTVGRG9Pfx1mc0TOAXhsEzsow0l0rT1i7isHBP6Vf60f6v1'
    'YQPyLsA8XyRROY+XCZi1/TYpchjV5s1dtw+7LZa1zIuQiN4++vmHbJXExUW+uoY/QLgtzC7ym8hVbE+xfcUOyLutXa6XWcUG'
    'zjvnCPR4m5U4qD9BpaKmwp0rpidNX5oBmdQmk4b0yN3nD/b5LcmTfG+ueH1pdvnevfw23Zd8X/L9uRLQ5fv38oMmPZD8QPID'
    'yQ8O8IOD/FNQZnd4E/2dBATqLrrM8+XOKjT3EgJMCDEhvCPBVXaEzxs2DB4/jdbZlhyhWfIt0eyM70EGNLFX8ZZI0zZ/SRab'
    'efJjvMXhJOW32q02dp6A8TpJ1ovsqnyqCbltFsqdotzpA+SGvHmGEqiUSw/KpVIulXLpo+RSKZeiXHpfOfiaG6ZYv/Qh9Qua'
    'hBATHlK/KW+eW2Yq65cerF8q65fK+qWPql8q65di/dIH1o/y5gwlUCmXHpRLpVwq5dJHyaVSLkW5d9WPAO4hXuxlFRLs7MFP'
    'ecWHUl8Bjts6vk6KKmMv7+gyLhOye2n3XxZgI21qmau8inC1SxOpdoOrb6QyJpUxnwlVIJO5wORNLZB19uDFasHDUhmGVOap'
    'w1IZ9ilgEqDTGmVlFG7Z5wZ7DHoh3yuuuqYn7M1XRkWyTuKKZUcu6Xjs4XdvNvFSRXigvH/3E7wOwusgPKkiOKDC7yD8BvES'
    'OgI7Hs96T/XEqwUT1XXhzLyC7p0O0e8S/S7RR+JXIKa+S/atd9plxWu0c4XZX8LuyoOdGOuIoRsHUS/qFeqC6rKO2UWaF9nb'
    'fFWx8N1LXIvOTgbshliDtesS3tT0073buD4Dy8jxlZiS1mpWcOtotsUIHUT0/8MNBTdsueE+N2y4VHBDwQ1r7he7Y8PvL6Pi'
    'B/iatJakCkez0UfoIKK/kxoKathSw31q2FCpoIaCilovgE8zNmJuRM/Y/HhyFZevSWvZI3Z6nMd7x71dyI2ACIH8rIKQxupA'
    'xJmlfQq0oZbJ2/otS6SJIzwH6VEP0coReDwv8vU6WZDGsIe/pUmRwNfQeEBnp3mXzc2mYt8XAthH63hhD17FC+d90K8Yz2aS'
    'VmUVr6pbbWCNK/Zgj1LniaFPxud6T+v1Zvy/QePQQNe5w3NOJpqt93r/fDNTjunOZNJ3tN5MSnU+NDTDZD+N3TG1/kAfjsaG'
    'ORNHf+VBQ8715YO0EXcESsSYO0IlwuCOZzJiWIudyoiRxh3PZcS4jqAywqgjzpwTw2AOgw+41yNkVs/e7x+Lz7P1AZwYmjWB'
    'vqGxH7DfR/x3+QmICa4j+t2ImQ69yfF/8yD+wW4NAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

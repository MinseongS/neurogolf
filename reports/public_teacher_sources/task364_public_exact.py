"""Task 364 public-teacher exact source draft.

Generated from `public_candidates/urad_7174_10/extracted/task364.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIACDpQGoC/52Vz2rbQBDGtSvZlicFOZs0dXpog9JCEb1Y/hdKS+wpxaA0EMgh0EtRbCUxdm03lhMf+yi59bn6Jp2xtIlK'
    'DCWSWGs98+2n3dWPWRs+/N6EfSgMJ7NFDOY87oEZTXpQCJc1v65k78ItnI6H/Qh2gf5QYOFan8N57JVBxtOqvBMSDjkFhf50'
    'cnOm5OScFNT1nsOzUXQ9icbf51fhLOqYHfNOlLxNsGbhYN4RyU0heAk0SpnDwfIf8yKbvwErDs+7wFklL7tusRfGV9G1twFW'
    'uBzOq8aDCrUK16s+Ahko86Rbc0vH4fJkOh0/YZZbwCNprQtlnpKFebwYwyfgPnv6uT197elnPH32rOf2rGvPesazzp6N3J4N'
    '7dnIeDbYs5nbs6k9mxnPJnu2cnu2tGcr49liz3YeT6YGaTTmpgY1NZihBpkazE0NamowQw0yNZibGtTUYIYaZGowNzWoqcEM'
    'NcjUYG5qUFODGWqQqcHc1KCmBjPUIFODuahRTGKbjdtKzBJLBWIGsn+gZHTgFr78XIRj2OGY1ff9prIi+tXxF0AikCP6zKO2'
    'krc1t3BGBY0L8UpHYfpatzVK+TpVoUCChbhI3pgt7cilHe9LO2ZLO1JpxzWlvQLigrILJW8I3e5gAO+Auqvl3lCQFnYSDrwt'
    'sH5MB5Fr0xkwj8NJfCdMWhnlQRyp4nQR0xzSlSlx6e3bZqWEfNwEVdNYf2kRHUdB1UqD2+lzR4verkTJmoKqSMMyfWpvr20L'
    'uk0SC0wOquBNkvp1mGq61KdmdNL46um9t/cqEldHULC3Zpb0Rsuh9zm24zgZNa5V2zQ1x7AMR7C6TFqiIRC2t8HDGIJA/Eni'
    'Iz8QMu3WA2Gm3XZAuN1vHwZV47/bhw/78mj7vto2iVZfM+gYT7zgYVWJ2y7tMfBO01TFUQBCmlahWLLL8O11CqLagW1bKCLV'
    'FtSA2itu53uQYrJSlB8r0AKjsvkXMVkzX7IIAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

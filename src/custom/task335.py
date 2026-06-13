"""Task 335 (ARC d4a91cb9): same hpwl rule as task246 with colors
red(2)/cyan(8) dots and a yellow(4) L-path. See _hpwl.build_hpwl."""

from ._hpwl import build_hpwl


def build(task):
    return build_hpwl(2, 8, 4)

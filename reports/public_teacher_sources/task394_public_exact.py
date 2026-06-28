"""Task 394 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task394.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAK78QGoC/41a65LbNBSOZcmXs5embgvF3HYy/MoM1y6UBX5sQwtMoAxTfjDDDzzexNt6m92U2KEdfvEEPENfhXfgMXgI'
    'JF8SWbtf5Uw0lo6+c/QdnSPJM1ZA0X6ZFk/vHB0m2Ytny1X5xd8P6QsS+cWzdUl7J4t09vRuUpTpqixop2lmF/Mi8upG3DxH'
    '4udFPsvoATUCEumLrDiMaqWkXJbpItYbo/BRNl/Psp/X5+NrFDzNsmfz/Ly4PXjpMLpDOlTaWmXpYUR5kVS15CTW6iPx4Pe1'
    'RF2ldLRROtKUjrZK91tnW75Vq+WrNV7J95B0KHlF/md2WI+tqnebsZt6O/bDdrIiWi2fy97HF3JQrd6O+TB9IcfkiuHx4Ng5'
    'dl86/mUS35GmGoWqns9ffHYYb6sj797qsbK2o6zlteJlSx/QViXym2rcVkb867QoxyGxcnnbU3jNkdlysXFkW0eOMOTIVjUK'
    'Vb1xZFPt78hGJfKbatxWLjuyaLNhKCHLVZHMklVWKGz0RkcySy/m+TwtZfU0xl2j4Nu0fJKtfrw/vk50kpazJ0nFzlGj/UZY'
    'M3oddMWoo+MNq+0jbLSrd8Sdlh6rnSZWVyfcXdKSmvzyySrLkpxE+XyZ5FHwLFvly3mSx5vaSPwiJyOjj6jNJdr0RdTUZE+s'
    '1Ufuw+VcaTRBu0JD9sRavdb4kjQjmtL+Vpqcrxex0ZbK6wXdI0NMmv1of5YWmZqqfL6WDsdGe+Tem8/p687keKfL9UrOjfdn'
    'tlKTs1/NeLI8PS2ysjLRabcT9SUZtskAqmW+LrMqrbfVmsERaZtktKuoNK087rQur4OvSNsqm4BSRycKqpYad1NrWT+gnZqK'
    '4l7QlldEp+vFIqnasVYfefUy6axm+oZ2/kgX+Tw5l0dUQZthokZcm9EbV9v5gXQMaeMSFdlFmV9kCxWSIltkszJrDRvt1rnv'
    'qbNayIBFVJynyn76/CjW6peoVSv0EWkQmWPpPJFG1AbkV4my/jzyakG8pzrLpVrEf6TFyP0pnY9vED9fzrNRMFteyEP6onzp'
    'uNFb7YleZ2w+S05yOf3zfCVpjj8N+NCfdE/26cHA8hvfqdT0N4DpgdN0tk/feI7fr5Tqg3U7RgtnzdNt4cOhN2nWx5RXkmtS'
    'UifflCv4+LoUtPvMlLsbrXp1TblSG9+UEi2wU75X23Im9XuEMv7X8VZwpATDe9KSM2lObiW5PxnvD9mkDcTUGYwfBYF0SAvT'
    '9Nj0yvZ703iO/3ODvcANXElaXzXTf11zslxt0pwrnq6G0+VIlxk6V+nqcjW5QhaviXEgSygLybIjy67Wz7V+X+snrV9o/YHW'
    'v/MK+zoHZF/ngOzrHJB9nQOyr3NA9nUO409UrGW0w4m+uU3f3OSP46h/VWl+438+DJwgDPYDJhX9yaV3k+nLD1H6mYuNgXR1'
    'LIvTsei5wD56usD+wGLf7cm/xXGAZwAvevrLLfPjADy3zLeJFwBn8hGAD4qHAHxQngjA523jieQMxJdZnrY8di15jXCsZ967'
    'Pfnb8Mg+B3jkp+jpL++5LhnIT2aZdw74ID9EzzgwkJ+2PBGAD8pDU+6C+LqWeNj2Qdu+5Vr0bPum25O/DY/sc4BH/ome/nLL'
    '/KB9nFv2ORfkJ1qXiLfb85zgPfNEgHFQHppyDuLLLesTnVus5zmJ/OOWcw7l0cBi3+3JH+WFbR8UPf3llvlB5zq3nKsc5Cc6'
    'B1Ae857vDbxnngigh/LQlAsQX2HZr9F7DLO8lw0s61tY3nvQvjWw2Hd78kf7kO1cFD395T3fW4Ul72znLjpXB5Z9TfR8j+Q9'
    '8wQ9UR6acg/E1wPjeiC+HshPD8TLA/npgXh5IN9s/G14ZJ8DPAN40dNfbpkfB+A5wDOAFyAuJh8B+DDARwA+KE8E4IPy0JT7'
    'IL4+GNcH8fVBfvogXj7ITx/Eywf5ZuNvwyP7HOAZwIue/nLL/DgAzwGeAbwAcTH5CMCHAT4C8EF5IgAflIemPADxDcC4AYhv'
    'APIzAPEKQH4GIF4ByDcbfxse2ecAzwBe9PSXW+bHAXgO8AzgBYiLyUcAPgzwEYAPyhMB+KA8NOUhiG8Ixg1BfEOQnyGIVwjy'
    'MwTxCkG+2fjb8Mg+B3gG8KKnv9wyPw7Ac4BnAC9AXEw+AvBhgI8AfFCeCMAH5aEp//Xd5mZB9BrdDJxoSCxwZCFZ3lHl5ICa'
    'L30Vgl1GnB1sbll0bajiq3J2q3MDJvKIS9jg7GbnQ7CShh3pkSa91bnJYphovmZvwLc7N06IAonlFZMb+g0SBfcl/Prms38l'
    '8moL2lUPw8L26oZmoRFuLNx51SWK7kSFsuzLws4+xjcjurO/VRl1v/tGEQ0lblfHnUXaPYOW3039HsIVUnWhYDsbxrUDvad7'
    'EaDT070S0Pbc0L+8t8LXjM/4rTzSvq+3srf0b+XRPu1KaSAddVU5e7vzWb3qDrXug0tfxk0D7+kfv6+Y9wo14TQY7v0P2FXc'
    'H6kmAAA='
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

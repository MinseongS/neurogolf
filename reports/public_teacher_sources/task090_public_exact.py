"""Task 090 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task090.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAJj8QGoC/6XdfW9k51nH8Xgfkt0TJ7sxJQQDBRn+QPvXXr/rOk9FNG0qUXXVAiICCkhYzu4oseJ6V7ZXifgLif95DbwC'
    'XgCvjnnw+PzO9TCxvK2izBx7zn12Zu7vue+P0vRRc/DeT/7rv+81f988PD1/8/aq2b88O325OL68Orm4umyazbPF+aubxyff'
    'Ly4PHn91dvLy2+Or128On2wO3xw4evjl6kAzNNMvHXy4efh2OLZXhx+9PLm8Ot4eOXrwi+XTZ4+be1evP7v3P3v3mquGf337'
    '2ovX3x0/5yfCT8BPlJ/Y4dPLN2en2wGXhy6Xl7g68uzD5sHJ96eXm1H/934zG6lZ/u358cvXZ7PHQo9Bj5UeGz1u6XFHj3t6'
    'PNDj8eDDaazn/ET4CfiJ8hPjJy0/6fhJz08GfsJXAL4C8BWArwB8BeArAF8B+ArAVwC+AoyHT9ZPNh/b8lD4wO6vPrC/augj'
    'at5/fb5YfmEONi+9eHt+/Obs7eWxHPoDR/d//upV85PGH2/ih7z6oRzS46P7v3l7djPw+lA1MPzAKAZGE79Rqx+CBkYcGNXA'
    '6gfWYmBt4td39UOlgTUOrNXA5ge2YmBr4lxZ/dBoYIsDWzVw6wdui4HbJk7M1Q9bGriNA7fVwJ0fuCsG7ppYgdUPOxq4iwN3'
    '1cC9H7gvBu6bmJzVD3sauI8D99XAgx94KAYemti31Q8HGniIAw/VwKMfeCwGHhuO6XTmkQYeNwP/NQ083gz81HXh+WE4shn6'
    'p034QZPEe12J54f8ZDP8Txs+Vo4vYXypxpcmuV+sTy88viTjSzk+wvioxkeT3KLWpwePj2R8lONrGF+r8bVJ7orr0yuPr8n4'
    'Wo5vYXyrxrcmuRGvT288viXjWzl+G8Zvq/HbJrn3r0/f8vhtMn5bjt+F8btq/K5Jlhvr03c8fpeM35Xj92H8vhq/b5IVzvr0'
    'PY/fJ+P35fhDGH+oxh+aZFG1Pv3A4w/J+EM5/hjGH6vxxyZZx61PP/L4YzJ+2T+E/qHqH7h/4P6B+4ekfyj7h9A/VP0D9w/c'
    'P3D/kPQPZf8Q+oeqf+D+gfsH7h+S/qHsH0L/UPUP3D9w/8D9Q9I/lP1D6B+q/oH7B+4fuH9I+oeyfwj9Q9U/cP/A/QP3D0n/'
    'UPYPoX+o+gfuH7h/4P4h6R/K/iH0D1X/wP0D9w/cPyT9Q9k/hP6h6h+4f+D+gfuHpH8o+4fQP1T9A/cP3D9w/3Ddv/+7P9tA'
    '8p6Ot1m88+HNCO8PeMnOq2he2PJac7bwm63CZkui2fpktliY3blnt9HZPW12g5nVfpbeWQdnUZoVYjZdZ3Nn9kWefatmHzF/'
    'BrTA/+701dU3l4efXn8Y5y9Pruj40fu/WB+ab/7nWrPZsAtpjdBGXmhvLbTdFdqBCm0KhfZpQlsnod2MkNYIL/iFV9/CS2Hh'
    'danwIlF4xSa8fBJeywgvLITv8sK3XOH7n/DNSPjOIJxp4WYKB0y4JsJTW1hr5HZaI4nWiNcaKbRGvNbMP+RJayRqjSRaI15r'
    'pNAa8Voz/0ZNWiNRayTRGvFaI4XWiNea+dd30hqJWiOJ1ojXGim0RrzWzOfKpDUStUYSrRGvNVJojXitmU/MSWskao0kWiNe'
    'a6TQGvFaM6/ApDUStUYSrRGvNVJojXitmSdn0hqJWiOJ1ojXGim0RrzWzPs2aY1ErZFEa8RrjRRaI15rhLRGSGskao0kWiNB'
    'a6TSGgla4+JNWiOJ1kimNRK0RiqtkaA17n5BWiOJ1kimNRK0RiqtkaA17hZFWiOJ1kimNRK0RiqtkaA17q5IWiOJ1kimNRK0'
    'RiqtkaA17kZMWiOJ1kimNRK0RiqtkaA17t5PWiOJ1kimNRK0RiqtkaA1brlBWiOJ1kimNRK0RiqtkaA1boVDWiOJ1kimNRK0'
    'RiqtkaA1blFFWiOJ1kimNRK0RiqtkaA1bh1HWiOJ1kimNRK0RiqtkaA1bulIWiOJ1kimNRK0RiqtkaA1brVKWiOJ1kimNRK0'
    'RiqtkaA1boFMWiOJ1kimNRK0RiqtkaA1bk1OWiOJ1kimNRK0RiqtkaA1bhtAWiOJ1kimNRK0RiqtkaA1budBWiOJ1kimNRK0'
    'RiqtkaA1brNDWiOJ1kimNRK0RiqtkaA1bn9FWiOJ1kimNRK0RiqtkaA1bktHWiOJ1kimNRK0RiqtkaA1wlojrDWSaY2Q1ghp'
    'jZDWCGmNkNYIaY2Q1ghpjZDWCGmNsNYIa42w1ghrjbDWCGuNsNYIa42w1ghrjbDWCGuNsNYIa42w1ghrjbDWCGuNsNYIa414'
    'rZG7ac1m3wzSGtBGHrS3Bm13QTtQ0KYQtE8DbZ1AuxmQ1oAX/ODVN3gpDF6XgheJ4BUbePkEXsuAFxbguzz4lgu+/4FvRuA7'
    'AzjT4GaCAwauCXhqg7UGt9MaJFoDrzUotAZea+Yf8qQ1iFqDRGvgtQaF1sBrzfwbNWkNotYg0Rp4rUGhNfBaM//6TlqDqDVI'
    'tAZea1BoDbzWzOfKpDWIWoNEa+C1BoXWwGvNfGJOWoOoNUi0Bl5rUGgNvNbMKzBpDaLWINEaeK1BoTXwWjNPzqQ1iFqDRGvg'
    'tQaF1sBrzbxvk9Ygag0SrYHXGhRaA681IK0BaQ2i1iDRGgStQaU1CFrj4k1ag0RrkGkNgtag0hoErXH3C9IaJFqDTGsQtAaV'
    '1iBojbtFkdYg0RpkWoOgNai0BkFr3F2RtAaJ1iDTGgStQaU1CFrjbsSkNUi0BpnWIGgNKq1B0Bp37yetQaI1yLQGQWtQaQ2C'
    '1rjlBmkNEq1BpjUIWoNKaxC0xq1wSGuQaA0yrUHQGlRag6A1blFFWoNEa5BpDYLWoNIaBK1x6zjSGiRag0xrELQGldYgaI1b'
    'OpLWINEaZFqDoDWotAZBa9xqlbQGidYg0xoErUGlNQha4xbIpDVItAaZ1iBoDSqtQdAatyYnrUGiNci0BkFrUGkNgta4bQBp'
    'DRKtQaY1CFqDSmsQtMbtPEhrkGgNMq1B0BpUWoOgNW6zQ1qDRGuQaQ2C1qDSGgStcfsr0hokWoNMaxC0BpXWIGiN29KR1iDR'
    'GmRag6A1qLQGQWvAWgPWGmRaA9IakNaAtAakNSCtAWkNSGtAWgPSGpDWgLUGrDVgrQFrDVhrwFoD1hqw1oC1Bqw1YK0Baw1Y'
    'a8BaA9YasNaAtQasNWCtAWsNvNbgblqz2b4qaY3SRl5pb6203VXagSptCpX2aUpbJ6XdjJLWKC/4lVffykth5XWp8iJRecWm'
    'vHxSXssoLyyU7/LKt1zl+5/yzUj5zqCcaeVmKgdMuSbKU1tZa/R2WqOJ1qjXGi20Rr3WzD/kSWs0ao0mWqNea7TQGvVaM/9G'
    'TVqjUWs00Rr1WqOF1qjXmvnXd9IajVqjidao1xottEa91sznyqQ1GrVGE61RrzVaaI16rZlPzElrNGqNJlqjXmu00Br1WjOv'
    'wKQ1GrVGE61RrzVaaI16rZknZ9IajVqjidao1xottEa91sz7NmmNRq3RRGvUa40WWqNea5S0RklrNGqNJlqjQWu00hoNWuPi'
    'TVqjidZopjUatEYrrdGgNe5+QVqjidZopjUatEYrrdGgNe4WRVqjidZopjUatEYrrdGgNe6uSFqjidZopjUatEYrrdGgNe5G'
    'TFqjidZopjUatEYrrdGgNe7eT1qjidZopjUatEYrrdGgNW65QVqjidZopjUatEYrrdGgNW6FQ1qjidZopjUatEYrrdGgNW5R'
    'RVqjidZopjUatEYrrdGgNW4dR1qjidZopjUatEYrrdGgNW7pSFqjidZopjUatEYrrdGgNW61SlqjidZopjUatEYrrdGgNW6B'
    'TFqjidZopjUatEYrrdGgNW5NTlqjidZopjUatEYrrdGgNW4bQFqjidZopjUatEYrrdGgNW7nQVqjidZopjUatEYrrdGgNW6z'
    'Q1qjidZopjUatEYrrdGgNW5/RVqjidZopjUatEYrrdGgNW5LR1qjidZopjUatEYrrdGgNcpao6w1mmmNktYoaY2S1ihpjZLW'
    'KGmNktYoaY2S1ihpjbLWKGuNstYoa42y1ihrjbLWKGuNstYoa42y1ihrjbLWKGuNstYoa42y1ihrjbLWKGuNeq3Ru2nNZhdp'
    'pDVGG3mjvbXRdtdoB2q0KTTapxltnYx2M0ZaY7zgN159Gy+FjdelxotE4xWb8fLJeC1jvLAwvssb33KN73/GNyPjO4Nxpo2b'
    'aRww45oYT21jrbHbaY0lWmNea6zQGvNaM/+QJ62xqDWWaI15rbFCa8xrzfwbNWmNRa2xRGvMa40VWmNea+Zf30lrLGqNJVpj'
    'Xmus0BrzWjOfK5PWWNQaS7TGvNZYoTXmtWY+MSetsag1lmiNea2xQmvMa828ApPWWNQaS7TGvNZYoTXmtWaenElrLGqNJVpj'
    'Xmus0BrzWjPv26Q1FrXGEq0xrzVWaI15rTHSGiOtsag1lmiNBa2xSmssaI2LN2mNJVpjmdZY0BqrtMaC1rj7BWmNJVpjmdZY'
    '0BqrtMaC1rhbFGmNJVpjmdZY0BqrtMaC1ri7ImmNJVpjmdZY0BqrtMaC1rgbMWmNJVpjmdZY0BqrtMaC1rh7P2mNJVpjmdZY'
    '0BqrtMaC1rjlBmmNJVpjmdZY0BqrtMaC1rgVDmmNJVpjmdZY0BqrtMaC1rhFFWmNJVpjmdZY0BqrtMaC1rh1HGmNJVpjmdZY'
    '0BqrtMaC1rilI2mNJVpjmdZY0BqrtMaC1rjVKmmNJVpjmdZY0BqrtMaC1rgFMmmNJVpjmdZY0BqrtMaC1rg1OWmNJVpjmdZY'
    '0BqrtMaC1rhtAGmNJVpjmdZY0BqrtMaC1ridB2mNJVpjmdZY0BqrtMaC1rjNDmmNJVpjmdZY0BqrtMaC1rj9FWmNJVpjmdZY'
    '0BqrtMaC1rgtHWmNJVpjmdZY0BqrtMaC1hhrjbHWWKY1RlpjpDVGWmOkNUZaY6Q1RlpjpDVGWmOkNcZaY6w1xlpjrDXGWmOs'
    'NcZaY6w1xlpjrDXGWmOsNcZaY6w1xlpjrDXGWmOsNcZaY6w15rXGbqs1nzf+33vT+P9p1UGz/vvxxfPVJn56vPz0T8+XGwc6'
    '1Ph/0IdeC3ot4mvReHai1yq9VuNrtfFvAr3W6LW2ee3n4c9YX7jQhUu8cNl14UIXLvHCZdeFC124zC8c/sLTwUGDIw6OXYOD'
    'Bsd8cPWDpydQOoFuT/Alf1UOPjp9fvzV4vJq89LD+dOjx/+wePX25eI3J99vvq+Ly58tv68fPHvSPPp2sXjz6vR3l5/trb7A'
    'v52d9JPtWRbnr45Pz18tvj+Mh47e//nF1zdnvp4J8cxfNPNrmrbUy8NvL0++OltcX7s/cPTBLy8WJ1eLi2b053i8fPrN4vTr'
    'b64OPlw+vDj57vhk+buH/GST1b9p4pWvh755tryWQ38g/tvLu8b/zs0fpDnd/MvGlj88pMeb28IvG76m5oP/WFy8Xr3ogI4u'
    'L+9q+cc8TI5N78GLxr8/TfL7Bx8vj/E53fPlRZ2/an7VuMPFRe5v37r1mzt7dvTwn79ZXCyW30Z/qumzmf9p1y9dYfbmh4fJ'
    'se1J/zaclN7X6axPtmfYvvv+wPZ8/xjO575PN6f8hM+w+VrGQ9OfncK7nIkyn4nyzjMRy+uROBPl7jNR8pkofibKjpko85ko'
    '00wUnomSzESJM1H8TJRbzESpZqLQTJT5TPx1w9fUzL7Nyy+oJNNRdk9H8dNRkukobjpKnI6rr7u46Vhe6f72TbyekxLn5N+F'
    '822/3/xFl2n2rf9vIQ7joe0J/z2ccPrYm2Qer9/OON+lmu//FM5Pn17jZ/X6S+PmvaTz/jic131749ye3hgugPxgAWRdAMwL'
    'gHcsgKwLgFgA3L0AyAsAXwDsKADKezG4AEgKgFgA+ALgFgVAVQBQARALAJ5XMisAkgJgdwHgC4CkAHAFQF4AuAKUV7q/fROv'
    'C4BYgN+G812/QU2c5NM3jFOAHSlAdb9Ppvj6fY0pwK4UwKUAlALxKYBPAcoUwKUA8xRITAFiCnCLxYAuU6DzFOg7LwZ0eT0a'
    'U6B3T4HmKVCfAt2RAp2nQKcUKKdAkxRoTIH6FOgtUqBVCpRSoDEFyhMMsxRokgLdnQL1KdAkBepSoHkK1KWgvNL97Zt4nQKN'
    'KfiXcL6bxUCc5dNXjFugO1qgrgU6tQBJCzRpge5qgboWKLUAvgXqW6BlC9S1QOctQGyBxhboLZYFqxbYvAX2zsuCVQsstsDu'
    '3gLLW2C+BbajBVZuDIxbYEkLLLbAfAvsFi2wqgVGLbDYAuMZprMWWNIC290C8y2wpAXmWmB5C8y1oLzS/e2beN0Cy5cFViwL'
    'NKbAYgpsRwqs3iFokgJLUmC7UmAuBUYpUJ8C8ymwMgXmUmDzFGhMgcUU2A+mAOsUtPMUtO+YAqxT0MYUtHdPQZunoPUpaHek'
    'oC13CC2noE1S0MYUtD4F7S1S0FYpaCkFbUxByxPMZilokxS0u1PQ+hS0SQpal4I2T0HrUlBe6f72TbxOQRtT8G/hfLSOj/N8'
    '+pJxDdodNWjrTYIlNWiTGrS7atC6GrRUA/M1aH0N2rIGratBO6+BxRq0sQbtLTYJtqxBN69B986bBFteTxdr0N29Bl1eg87X'
    'oNtRg25eg26qQcc16JIadLEGna9Bd4sadFUNOqpBF2vQ8RxrZzXokhp0u2vQ+Rp0SQ06V4Mur0HnalBe6f72TbyuQZdvErpq'
    'k9DGFnSxBd2OFnSuBd3UgjZpQZe0oNvVgs61oKMWtL4FnW9BV7agcy3o5i1oYwu62ILuFpuEVQv6eQv6d94krFrQxxb0d29B'
    'n7eg9y3od7SgL8Gg5xb0SQv62ILet6C/RQv6qgU9taCPLeh5hnWzFvRJC/rdLeh9C/qkBb1rQZ+3oHctKK90f/smXregzzcJ'
    'fbFJ6GIK+piCfkcK+toLuiQFfZKCflcKepeCnlLQ+RT0PgV9mYLepaCfp6CLKehjCvpbbBJWKRjmKRjeeZOwSsEQUzDcPQVD'
    'noLBp2DYkYKh9IKBUzAkKRhiCgafguEWKRiqFAyUgiGmYOAJ1s9SMCQpGHanYPApGJIUDC4FQ56CwaWgvNL97Zt4nYIh3yQM'
    '9SahjzUYYg2GHTUYajLokxoMSQ2GXTUYXA0GqkHvazD4GgxlDQZXg2Fegz7WYIg1GH6wBrquwTivwfiONdB1DcZYg/HuNRjz'
    'Goy+BuOOGowlGYxcgzGpwRhrMPoajLeowVjVYKQajLEGI8+xYVaDManBuLsGo6/BmNRgdDUYYw1Ws3esJlicl9OXgmfvuGP2'
    'jvUWf0hm75jM3nHX7B3d7B1p9g5+9o5+9o7l7B3d7B3ns3eIs3eMs3csZu+vGj90E3/54OOb55s32j0/uv/l26+aoXm0emO+'
    'vjilc0yf1+PVw1eLs6uTw+nh5pVjMx1pknf5YH3i351cfnt48+jowa8Xl5dN2zxaXcZ6UHdVB49XD69HvHm4vdbpSPYHXp90'
    'M+D20fWA0txcQnPzs4Pmzcnp+dXmFfR4883+vKFDy8en598evzn9fnHWPDw9f/P26uD912+vln8/fLL5tYvFy6uT86/PFtcf'
    '0sGfXy1f+Hzc/AO7ZycXX68u9vp/br393Wd/8eje0w++2L88O3252LwBly+evuf+8+xo/VvN5reWn/byd/auf/Yw/Z1VpKff'
    'ubf9nY+f3vtiu+l+sffes4+Wz68D9GJv75k92lv+98eP9paHb74VL3783t69+w8evv/Bo8fNh/sfffzk6ScHv/ej3//0Dz77'
    'w8M/+uM/uX7V8nWrV20/1h981c+Wr2hWr3u69wW9uS/+8r3yP//5+exNebocb+rBi+VtY3NEbo7cvz6iN0ceXB/pbo48/Nc/'
    '3X6gnzY/erR38LS592hv+Vez/OvHq7+++rPm+qOufuOLB817Tz/+f0YZTTARhwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

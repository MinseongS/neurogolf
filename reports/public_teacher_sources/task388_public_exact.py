"""Task 388 public-teacher exact source draft.

Generated from `public_candidates/biohack_mix_20260628/_src_A/task388.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    'H4sIAK78QGoC/81X627jRBSOc3Gck7TNDqV0p7vsYpZbEBA7TpqCuLTLAjILQhQJCYRG3sbdpnWdbOx0K8RD7CPwgwfjUTgz'
    'vs3YXST+YcnynO9cfGbmzMw3Bnz89zvwPbTm4XIdw9bJYh3GkTVk3rUfsQPSzQFrSGXB7Pzoz9Yn/vH6crAFxoXvL2fzy2i3'
    '9qdWh8cgm8JGvIi9IAlpWcRIRMumeetfo70LuR2BkJ0GCy9m1ohKbbN5/GwVw0cgYaQdsvWUWQ7NGmbzoRfFgw7U48VunYd+'
    'HzJd3sD0ni9YyKwxzVtm43A2gz/UXu2cLILFiiUQi2JvhR9rAtsK7oczRPdfZk22FNwe0jJgto6D+YkPP0NZQ/oJ4K2eMtti'
    '84lDezJi6oerp99514MuNL3rebSrYZerw2tBJQ5R48jD1uIuh6AYAPiBf+WHbM6cokMzH2XbpmUgGcwvoIyTDQmwR1QVq3P3'
    'VVa0/SeBd3IxyQbVdmAzRcTg22PSSWV7SotmNrBjKDDQz7zglFmkFy5iJmBmH1BFMpuP/SiCD8BYLZ5HEzYaSrUzD9kKIYvm'
    'rcIc+8PNbdX8BKERzVuFeRYAch0xvJXvYcuheQtHM5zBCJQUIVeTznIxD2OHjca0aCZOn0GBAERn3tIfoc+EbAgYFxKuo9E+'
    'VUWz/aMvbGEKqoZAKrLRlEptZeqAT90ZSGpo4dA7FsFdYsmuvCDi6AHpcXE+u544zBnSJkoXZvOnxfLbvJx59Q42oR1gFfpR'
    'nJT3BujRYhX7s6S4P4ZS2G4uOjaVhWqBfQpKDkmCKPHJc7A+FbHq/iGoFmBEc9EcEx0rhzkOTb9m48v5FbwHqSgZthHBwRnT'
    'rGE2vlsHON2l0JmadPhcDIfM2adFM1lx0l7XjZ/7wZXPpQnRQ8tmzpSm3+QPH0LhDqkmCY6bn3NAi2YS/CPZPq/vBLPYeEiL'
    'puIgQpQcLLSyaNFMHDz5D5JvERgKF7IpajIdoLFNS7KpP1yEJ16s1BJ8AiUz0s3l8YjKgjLdOnf+HORqAnX3Ihvr5YwJaMLG'
    'DlXFZMh9UNH/JBIDRdthYzy1stbNnYwBfvdXC8tBkwnIfYLcM9uKhXK8T1XR3DrGqLG/ehT4lz6eReqKfAU6K36Yx/NFaDYu'
    'ves/tQY8BDUGbIg9xLKvsbTG0+zAufSWbHxAFanYcH4FRSHOP5Yh12wypBUkIxb5MehHX2CW7eox+DVUnPFcmz89i3ktjEjf'
    'wx5dYcbJnE4sWkFwHuchP0/4wYDwxIacQpAuX9x8W2YTLCVJSHd8dOMHBHdzZDeeVGI5prKQun0GMkg2c4Gv7QktydVd6hAq'
    '3YCSU5LDEy9C5T6VBeywdw3fgNwdkA2gw2uNh7FJT2DJP6ZUkczWz2f+yocDdYJBMSJwEniRGKADKrWTJH4DCYLe6TxEwrj0'
    '8PyfyklsCCN2ug4Ctj+kqmg2fvBmWL/NS/yjied1iJQijHn9YsEqpshezrww9AORXcT2ib5Yx8hGaPo1W4+erb2A7MRedDGa'
    'Tlm09FbYmShZOoN+XztKyYbbrNVqnyNSPyoydbXa4BYiUgm6mjHYNrR+60jiW269WxtsoWF+aLiaPiAIyJu8q/UG7xiaAfhq'
    'qCtn74LW7W1sbvVvGa8MHKPZbx8pQ+jer5UeUvpi+Dp6lS8Qbr+eGjQyw7eEoXoncPtZXC0zexsTbR+9hDa7Rm73QNjdSLpd'
    'AzKrqehThSlW+1V+BhPhWWKU7n2tlK9e+g52DV0Mdc4SXb2m1RvN1mDH0Dme0cEcp6InEhFzjQdZtFeFLmFKrpH/5C/NeMF/'
    'Im3p7gut9j97silXtny330vV2Xdwz+iJISv2T7eXjI3eNjrQHdw1elyd75Oq+pd76Y2A7ACuE9KHuqHhC/i+zt8n9yFdncKi'
    'U7U4v6vc8cgmYEbEyMzOiXQV1aGJutr5tnLpzNBbBanhUB0hIm3rGfZG9Van/rNz/voN9zMAw2iTJrc5p+plTOhaqe5u9ZYl'
    'q/fKNIUr66nyTeleVBpR/ur8PX9bvXiUxrWwo8V9RnSwIzqoix/tSDccnkBHJCB88muM6qPz5PK7y0t/ule5oOTRH5zvyncQ'
    'oYFUs1e+MxRKnpR6KShmQj+/rVBAaTD1LGbO2BXldsb8FfTVgtXL8GsSD1YU/ZynZ8X1msSTb4zBWfNNioRDy4o7FWpcaHd5'
    'xyUeKVR6qtorE1U56o5EOeV4d0pskXShg8oWNHC7wwWh0kBeG3VRGz3+pgtGYXJSdKEvUx5Ff1fhM1Lp9US53lYZV1FTPT5K'
    'Jeokx72tUKNSSirVKXfpjsxuKtp7JXIiGYhd7agJtX7/H8VTceTrEwAA'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))

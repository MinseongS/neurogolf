import os
import onnx

def build(task):
    return onnx.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regime.onnx'))

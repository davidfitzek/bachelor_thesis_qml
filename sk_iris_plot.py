# sk_iris_plot.py

from pennylane import numpy as np

import matplotlib.pyplot as plt

import json
import lzma

filename = 'data/sk_iris_result.json.xz'
doc = {}

with lzma.open(filename, 'r') as f:
    filedata = f.read()
    doc = json.loads(filedata)


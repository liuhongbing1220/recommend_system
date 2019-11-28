import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os
os.makedirs('./saved_pb', exist_ok=True)
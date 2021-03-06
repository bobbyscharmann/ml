import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    first = random_matrix = tf.constant([1, 2, 3, 4])
    second = random_matrix = tf.constant([1, 2, 3, 4])
    dot_operation = tf.multiply(first, second)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(dot_operation)
    print("result is: ", result)

# It can be hard to see the results on the terminal with lots of output -- add 
# some newlines to improve readability.
print("\n" * 5)
print("Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)

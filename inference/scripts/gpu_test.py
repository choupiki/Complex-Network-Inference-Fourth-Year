# import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import numpy as np
# import tensorflow as tf
# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# # tf.debugging.set_log_device_placement(True)
# with tf.device("/cpu:0"):
#     x = np.array([1, 2, 3, 4])
#     print(x)
#     # Create some tensors
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     print(a)
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     print(b)
#     c = tf.matmul(a, b)
#     print(c)

import tensorflow as tf
import time
import matplotlib.pyplot as plt


cpu_times = []
gpu_times = []
sizes = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
for size in sizes:
    tf.compat.v1.reset_default_graph()
    start = time.time()
    with tf.device('cpu:0'):
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(op)
    cpu_times.append(time.time() - start)
    # print('cpu time took: {0:.4f}'.format(time.time() - start))
for size in sizes:
    tf.compat.v1.reset_default_graph()
    start = time.time()
    with tf.device('gpu:0'):
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(op)
    gpu_times.append(time.time() - start)
plt.plot(sizes, gpu_times)
plt.plot(sizes, cpu_times)
plt.show()
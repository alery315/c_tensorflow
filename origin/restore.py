from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import numpy as np

print(tf.VERSION)


pb_file_path = os.getcwd()
sess = tf.Session()
with gfile.FastGFile('model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

input_x = sess.graph.get_tensor_by_name('actor/actor_inputs/X:0')

op = sess.graph.get_tensor_by_name('actor/NN_output/Softmax:0')
t = np.ones((7, 16))
x = np.reshape(t, (-1, 7, 16))
ret = sess.run(op, feed_dict={input_x: x})
print(ret)

from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import numpy as np 

pb_file_path = os.getcwd()
sess = tf.Session()
with gfile.FastGFile('model2.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())


input_x = sess.graph.get_tensor_by_name('actor_inputs/x:0')

op = sess.graph.get_tensor_by_name('NN_output/FullyConnected/Relu:0')
t = np.ones((2,3))
x = np.reshape(t,(-1,2,3))
ret = sess.run(op,  feed_dict={input_x: x})
print(ret)
# [[0.         0.24045381 0.         0.         0.         0.        ]]

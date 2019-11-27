import tensorflow as tf
import tflearn
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()  # 获取当前代码路径
print(pb_file_path)

with tf.Session(graph=tf.Graph()) as sess:  # 代码结构一定要在with里面写，不能写在with上面
    with tf.variable_scope('actor_inputs'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='x')
        b = tf.Variable([[1., 2., 3.], [3., 2., 1.]], name='b')
    # xy = tf.multiply(x, y)
    with tf.variable_scope('NN_output'):
        op = tf.add(x, b, name='op_to_store')  # 输出名称，此处是成败的关键
        # op = tflearn.fully_connected(op, 6, activation='relu')

    sess.run(tf.global_variables_initializer())
    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['actor_inputs/x','NN_output/FullyConnected/Relu'])  # 此处务必和前面的输入输出对应上，其他的不用管
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['actor_inputs/x','NN_output/op_to_store'])  # 此处务必和前面的输入输出对应上，其他的不用管

    # 测试 OP
    feed_dict = {x: [[[10., 1., 2.], [10., 1., 2.]]]}
    print(sess.run(op, feed_dict))

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_file_path + '/alery.pb', mode='wb') as f:  # 模型的名字是model.pb
        f.write(constant_graph.SerializeToString())

    # 输出
    # INFO:tensorflow:Froze 1 variables.
    # Converted 1 variables to const ops.
    # [[0.03883655 0.64909285 0.         0.         0.         0.        ]]

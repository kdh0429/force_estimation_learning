import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input = 28
num_output = 6
output_idx = 6

# raw data
f = open('raw_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
t = []
x_data_raw = []
y_data_raw = []

for line in rdr:
    line = [float(i) for i in line]
    t.append(line[0])
    x_data_raw.append(line[1:num_input+1])
    y_data_raw.append(line[-num_output:])
    #y_data_raw.append(line[-output_idx])

t = np.reshape(t,(-1,1))
x_data_raw = np.reshape(x_data_raw, (-1, num_input))
y_data_raw = np.reshape(y_data_raw, (-1, num_output))

tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
new_saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("m1/input:0")
y = graph.get_tensor_by_name("m1/output:0")
keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
hypothesis = graph.get_tensor_by_name("m1/hypothesis:0")

hypo = sess.run(hypothesis, feed_dict={x: x_data_raw, keep_prob: 1.0})


res_idx = 0
mean_error = np.mean(np.abs(y_data_raw[:,res_idx]-hypo[:,res_idx]))
print("Mean error : %f" % mean_error)
 

# plt.plot(t,y_data_raw[:,res_idx], 'r', label='real')
# plt.plot(t,hypo[:,res_idx], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('Fx')
# plt.legend()
# plt.show()


plt.subplot(3,1,1)
plt.plot(t,y_data_raw[:,res_idx], 'r', label='real')
plt.plot(t,hypo[:,res_idx], 'b', label='prediction')
plt.xlabel('time')
plt.ylabel('Fx')
plt.legend()
plt.subplot(3,1,2)
plt.plot(t,y_data_raw[:,1], 'r', label='real')
plt.plot(t,hypo[:,1], 'b', label='prediction')
plt.xlabel('time')
plt.ylabel('Fy')
plt.legend()
plt.subplot(3,1,3)
plt.plot(t,y_data_raw[:,2], 'r', label='real')
plt.plot(t,hypo[:,2], 'b', label='prediction')
plt.xlabel('time')
plt.ylabel('Fz')
plt.legend()
plt.show()
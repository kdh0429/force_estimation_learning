import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# parameters
num_input = 4
num_output = 1
output_idx = 0 # num_input + num_output

# raw data
f = open('raw_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
t = []
x_data_raw = []
y_data_raw = []

for line in rdr:
    line = [float(i) for i in line]
    t.append(line[0])
    x_data_raw.append(line[1+output_idx])
    x_data_raw.append(line[8+output_idx])
    x_data_raw.append(line[15+output_idx])
    x_data_raw.append(line[22+output_idx])
    y_data_raw.append(line[29+output_idx])
    #x_data_raw.append(line[1:num_input+1])
    # y_data_raw.append(line[num_input+1:output_idx+1])

t = np.reshape(t,(-1,1))
x_data_raw = np.reshape(x_data_raw, (-1, num_input))
#x_data_raw = preprocessing.scale(x_data_raw)
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


plt.subplot(1,1,1)
plt.plot(t,y_data_raw[:,0], 'r', label='real')
plt.plot(t,hypo[:,0], 'b', label='prediction')
plt.xlabel('time')
plt.ylabel('joint0')
plt.legend()
plt.show()


# plt.subplot(2,4,1)
# plt.plot(t,y_data_raw[:,0], 'r', label='real')
# plt.plot(t,hypo[:,0], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint1')
# plt.legend()
# plt.subplot(2,4,2)
# plt.plot(t,y_data_raw[:,1], 'r', label='real')
# plt.plot(t,hypo[:,1], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint2')
# plt.legend()
# plt.subplot(2,4,3)
# plt.plot(t,y_data_raw[:,2], 'r', label='real')
# plt.plot(t,hypo[:,2], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint3')
# plt.legend()
# plt.subplot(2,4,4)
# plt.plot(t,y_data_raw[:,3], 'r', label='real')
# plt.plot(t,hypo[:,3], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint4')
# plt.legend()
# plt.subplot(2,4,5)
# plt.plot(t,y_data_raw[:,4], 'r', label='real')
# plt.plot(t,hypo[:,4], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint5')
# plt.legend()
# plt.subplot(2,4,6)
# plt.plot(t,y_data_raw[:,5], 'r', label='real')
# plt.plot(t,hypo[:,5], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint6')
# plt.legend()
# plt.subplot(2,4,7)
# plt.plot(t,y_data_raw[:,6], 'r', label='real')
# plt.plot(t,hypo[:,6], 'b', label='prediction')
# plt.xlabel('time')
# plt.ylabel('joint6')
# plt.legend()
# plt.show()
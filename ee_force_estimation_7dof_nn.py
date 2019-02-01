import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import wandb
import os

wandb_use = True
start_time = time.time()
if wandb_use == True:
    wandb.init(project="dusan_ws", tensorboard=False)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, shape=[None, num_input], name = "input")
            self.Y = tf.placeholder(tf.float32, shape=[None, num_output], name= "output")

            # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # weights & bias for nn layers
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W1 = tf.get_variable("W1", shape=[num_input, 10], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([10]), name='b1')
            L1 = tf.matmul(self.X, W1) +b1
            L1 = tf.nn.relu(L1)
            #L1 = tf.nn.sigmoid(L1)
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            W2 = tf.get_variable("W2", shape=[10, 10], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([10]), name='b2')
            L2 = tf.matmul(L1, W2) +b2
            L2 = tf.nn.relu(L2)
            #L2 = tf.nn.sigmoid(L2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            W3 = tf.get_variable("W3", shape=[10, 10], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([10]), name='b3')
            L3 = tf.matmul(L2, W3) +b3
            L3 = tf.nn.relu(L3)
            #L3 = tf.nn.sigmoid(L3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            W4 = tf.get_variable("W4", shape=[10, 10], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([10]), name='b4')
            L4 = tf.matmul(L3, W4) +b4
            L4 = tf.nn.relu(L4)
            #L4 = tf.nn.sigmoid(L4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.get_variable("W5", shape=[10, num_output], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([num_output]), name='b5')
            self.hypothesis = tf.matmul(L4, W5) + b5
            self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.abs(self.hypothesis - self.Y))
            #self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypothesis - self.Y)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost)

        self.mean_error = tf.reduce_mean(tf.abs(self.Y-self.hypothesis))

    def get_mean_error_hypothesis(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run([self.mean_error, self.hypothesis, self.X, self.Y], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=1.0):
        return self.sess.run([self.mean_error, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

    def next_batch(self, num, data):
        x_batch = []
        y_batch = []
        i = 0
        for line in data:
            line = [float(i) for i in line]
            x_batch.append(line[1:num_input+1])
            y_batch.append(line[-num_output:])
            #y_batch.append(line[-output_idx])
            i = i+1

            if i == num:
                break
        return [np.asarray(np.reshape(x_batch, (-1, num_input))), np.asarray(np.reshape(y_batch,(-1,num_output)))]
# input/output number
num_input = 28
num_output = 6
output_idx = 6
# loading testing data
f = open('testing_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
t = []
x_data_test = []
y_data_test = []

for line in rdr:
    line = [float(i) for i in line]
    t.append(line[0])
    x_data_test.append(line[1:num_input+1])
    y_data_test.append(line[-num_output:])
    #y_data_test.append(line[-output_idx])

t = np.reshape(t,(-1,1))
x_data_test = np.reshape(x_data_test, (-1, num_input))
y_data_test = np.reshape(y_data_test, (-1, num_output))

# load validation data
f = open('validation_data_.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
x_data_val = []
y_data_val = []
for line in rdr:
    line = [float(i) for i in line]
    x_data_val.append(line[1:num_input+1])
    y_data_val.append(line[-num_output:])
    #y_data_val.append(line[-output_idx])
x_data_val = np.reshape(x_data_val, (-1, num_input))
y_data_val = np.reshape(y_data_val, (-1, num_output))

# parameters
learning_rate = 0.005
training_epochs = 1000
batch_size = 100
total_batch = int(np.shape(x_data_test)[0]/batch_size*7)
drop_out = 1.0

if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.total_batch = total_batch
    wandb.config.activation_function = "ReLU"
    wandb.config.training_episode = 440

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

train_mse = np.zeros(training_epochs)
validation_mse = np.zeros(training_epochs)

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    f = open('training_data_.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)

    for i in range(total_batch):
        batch_xs, batch_ys = m1.next_batch(batch_size, rdr)
        c, _ = m1.train(batch_xs, batch_ys, drop_out)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    [cost, hypo, x_test, y_test] = m1.get_mean_error_hypothesis(x_data_val, y_data_val)
    print('Validation cost:', '{:.9f}'.format(cost))

    train_mse[epoch] = avg_cost
    validation_mse[epoch] = cost

    if wandb_use == True:
        wandb.log({'training cost': avg_cost, 'validation cost': cost})

        if epoch % 20 ==0:
            for var in tf.trainable_variables():
                name = var.name
                wandb.log({name: sess.run(var)})

print('Learning Finished!')


[error, hypo, x_test, y_test] = m1.get_mean_error_hypothesis(x_data_test, y_data_test)
# print('Error: ', error,"\n x_data: ", x_test,"\nHypothesis: ", hypo, "\n y_data: ", y_test)
print('Test Error: ', error)


elapsed_time = time.time() - start_time
print(elapsed_time)


saver = tf.train.Saver()
saver.save(sess,'model/model.ckpt')

if wandb_use == True:
    wandb.save(os.path.join(wandb.run.dir, 'model/model.ckpt'))
    wandb.config.elapsed_time = elapsed_time

epoch = np.arange(training_epochs)
plt.plot(epoch, train_mse, 'r', label='train')
plt.plot(epoch, validation_mse, 'b', label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('abs error')
plt.show()

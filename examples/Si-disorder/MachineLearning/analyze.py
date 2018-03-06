#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from numpy.random import randint
from matplotlib.pylab import *
import deepdish as dee
import os

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def get_next_batch(conf,y,n):

 x_set = []
 y_set = []
 N = len(y)
 for i in range(n):
  k = randint(0,N)
  x_set.append(conf[k])
  y_set.append(y[k])


 return np.array(x_set),np.array(y_set)

kappa = np.load('kappa.dat')
conf = np.load('training.dat')


dirs = os.listdir('../confs')
kappa = []
confs = []
for dd in dirs:
 if dd.split('_')[0] == 'kappa':
  a = float(np.load('../confs/' + dd))
  if a > 0.0: 
    conf = np.load('../confs/conf_' + str(dd.split('_')[1]))
    confs.append(conf.flatten()/20.0)
    kappa.append(a/145.0)
conf = np.array(confs)


N = len(kappa)
Np = len(conf[0])


conf_test = []
conf_training = []
y_test = []
y_training = []

mm = np.mean(kappa)
for n in range(N):
 if n < 600:
  conf_training.append(conf[n])
 else:
  conf_test.append(conf[n])

 if kappa[n] < mm:
  if n < 600:
   y_training.append([1,0])
  else:
   y_test.append([1,0])
 else:
  if n < 600:
   y_training.append([0,1])
  else:
   y_test.append([0,1])


#plot(range(N),kappa,ls='none',marker='o')
#plot([0,100],[np.mean(kappa),np.mean(kappa)])
#xlabel('kappa')
#ylabel('Configuration')
#show()
#quit()
x = tf.placeholder(tf.float32, [None,32])

#W = tf.Variable(tf.random_normal([32, 32], stddev=0.1)) 
#W1 = tf.Variable(tf.random_normal([32, 32], stddev=0.1))
#W2 = tf.Variable(tf.random_normal([32, 2], stddev=0.1))

W = tf.Variable(tf.zeros([32, 32]))
W1 = tf.Variable(tf.zeros([32, 32]))
W2 = tf.Variable(tf.zeros([32, 2]))
b = tf.Variable(tf.zeros([32]))
b1 = tf.Variable(tf.zeros([32]))
b2 = tf.Variable(tf.zeros([2]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b
y = tf.nn.relu(y)
y = tf.matmul(y, W1) + b1
y = tf.nn.relu(y)
y = tf.matmul(y, W2) + b2



#Known
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = get_next_batch(conf_training,y_training,100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: conf_test, y_: y_test}))



#from matplotlib.pylab import *

#plot(range(N),kappa,linestyle='none',marker='o')
#show()



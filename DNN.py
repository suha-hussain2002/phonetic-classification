#Dependencies
from __future__ import division
from __future__ import print_function
from random import randint
import os
import tensorflow as tf
import sys
import urllib
import numpy as np

#Global variables
hidden_neurons = 600
learning_rate = 0.05
batch_size = 200
w_stddev = 0.01
b_constant = 0.001
step_number = 2000000
LOGDIR = '/home/sh4976/speech_data/tb/'

#Loading datasets
test_data = np.load('/home/sh4976/speech_data/mydata/NORM_TEST_DATA.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')
test_target = np.load('/home/sh4976/speech_data/mydata/1TEST_TARGET.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')
valid_data = np.load('/home/sh4976/speech_data/mydata/NORM_VALID_DATA.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')
valid_target = np.load('/home/sh4976/speech_data/mydata/1VALID_TARGET.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')
train_data = np.load('/home/sh4976/speech_data/mydata/NORM_TRAIN_DATA.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')
train_target = np.load('/home/sh4976/speech_data/mydata/1TRAIN_TARGET.npy', mmap_mode='r', allow_pickle = False, fix_imports=False, encoding='ASCII')

#Create batches of data and targets to train upon
def make_batch(data, target, batch_size):
    rand_array = np.random.randint(1059099, size=batch_size)
    batch_data = data[rand_array, :]
    batch_target = target[rand_array, :]
    return batch_data, batch_target

#The definition of a fully connected layer
def fc_layer(input, size_in, size_out, keep_prob, name):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=w_stddev), name="W")
    b = tf.Variable(tf.constant(b_constant, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    act_drop = tf.nn.dropout(act, keep_prob)
    #Create histograms of relevant values
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    tf.summary.histogram("activations after dropout", act_drop)
    return act_drop

#The last layer of the network
def output_layer(input, size_in, size_out, name):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=w_stddev), name="W")
    b = tf.Variable(tf.constant(b_constant, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b) #no dropout for the output
    #create histograms of relevant values
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("act", act)
    return act

#Model definition
def dnn(learning_rate):
  #Reset graph and start session
  tf.reset_default_graph()
  sess = tf.Session()

  with tf.name_scope("inputs"):
      # Setup placeholders
      x = tf.placeholder(tf.float32, name="data")
      y = tf.placeholder(tf.float32, name="target")
      keep_prob = tf.placeholder(tf.float32, name="kprob")
      #Create histograms of the inputs
      tf.summary.histogram("x", x)
      tf.summary.histogram("y", y)

  with tf.name_scope("fc_layerss"):
      #The architecture of the model
      fc1 = fc_layer(x, 1845, hidden_neurons, keep_prob, "fc1")
      tf.summary.histogram("fc1", fc1)

      fc2 = fc_layer(fc1, hidden_neurons, hidden_neurons, keep_prob, "fc2")
      tf.summary.histogram("fc2", fc2)

      fc3 = fc_layer(fc2, hidden_neurons, hidden_neurons, keep_prob, "fc3")
      tf.summary.histogram("fc3", fc3)

      fc4 = fc_layer(fc3, hidden_neurons, hidden_neurons, keep_prob, "fc4")
      tf.summary.histogram("fc4", fc4)

      logits = output_layer(fc4, hidden_neurons, 39, "5") #39
      tf.summary.histogram("logits", logits)


  with tf.name_scope("xent"):
      #Cross Entropy/Loss calculation
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)
    tf.summary.histogram("xent", xent)

  with tf.name_scope("train"):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"):
      #Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("accuracy", accuracy)

  summ = tf.summary.merge_all()

  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR)
  writer.add_graph(sess.graph)

  for i in range(step_number):
      #Train
    batch_data, batch_target = make_batch(train_data, train_target, batch_size)
    sess.run(train_step, feed_dict={x: batch_data , y: batch_target, keep_prob: 0.9})
    if i % 1000 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch_data , y: batch_target, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy)) #Prints training accuracy every 1000 steps
      writer.add_summary(s, i)
    if i % 3000 == 0:
        #Validate
      [valid_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:valid_data , y: valid_target, keep_prob: 1.0 })
      print('step %d, valid accuracy %g' % (i, valid_accuracy))
      saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
  #Test the model
  [test_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:test_data , y: test_target, keep_prob: 1.0})
  print('test accuracy %g' %test_accuracy)

def main():
    dnn(learning_rate)

if __name__ == '__main__':
  main()

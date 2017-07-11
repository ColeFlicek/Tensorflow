import tensorflow as tf

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
tf.summary.histogram("Weights", W)
tf.summary.histogram("Bias", b)

with tf.Session() as sess:
    # Init the graph
    sess = tf.global_variables_initializer()
    y = tf.matmul(x, W) + b

    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y))
    tf.summary.scalar("Loss", cross_entropy)

    with tf.name_scope("Train_Model"):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        for i in range(1000):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
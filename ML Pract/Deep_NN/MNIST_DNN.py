import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Build model

n_nodes_h11 = 500
n_nodes_h12 = 500
n_nodes_h13 = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def Neural_Network_model(data):

    hidden_1_layer = {'Weights': tf.Variable(tf.random_normal([784, n_nodes_h11])),
                      'Biases': tf.Variable(tf.random_normal([n_nodes_h11]))}

    hidden_2_layer = {'Weights': tf.Variable(tf.random_normal([n_nodes_h11, n_nodes_h12])),
                      'Biases': tf.Variable(tf.random_normal([n_nodes_h12]))}

    hidden_3_layer = {'Weights': tf.Variable(tf.random_normal([n_nodes_h12, n_nodes_h13])),
                      'Biases': tf.Variable(tf.random_normal([n_nodes_h13]))}

    output_layer = {'Weights': tf.Variable(tf.random_normal([n_nodes_h13, n_classes])),
                      'Biases': tf.Variable(tf.random_normal([n_classes]))}

    Layer_1 = tf.add(tf.matmul(data, hidden_1_layer['Weights']), hidden_1_layer['Biases'])
    Layer_1 = tf.nn.relu(Layer_1)

    Layer_2 = tf.add(tf.matmul(Layer_1, hidden_2_layer['Weights']), hidden_2_layer['Biases'])
    Layer_2 = tf.nn.relu(Layer_2)

    Layer_3 = tf.add(tf.matmul(Layer_2, hidden_3_layer['Weights']), hidden_3_layer['Biases'])
    Layer_3 = tf.nn.relu(Layer_3)

    output = tf.add(tf.matmul(Layer_3, output_layer['Weights']), output_layer['Biases'])

    return output

def Train_NNet(x):
    prediction = Neural_Network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epochs in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):

                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={ x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epochs, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

Train_NNet(x)
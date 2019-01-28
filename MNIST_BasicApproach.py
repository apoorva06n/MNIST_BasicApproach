import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def main():
	mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)
	plt.imshow(mnist.train.images[1].reshape(28,28))
	plt.show()
	x = tf.placeholder(tf.float32,shape=[None,784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	# Create the Graph
	y = tf.matmul(x,W) + b 
	y_true = tf.placeholder(tf.float32,[None,10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
	train = optimizer.minimize(cross_entropy)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		# Train the model for 1000 steps on the training set
		# Using built in batch feeder from mnist for convenience

		for step in range(1000):
			batch_x , batch_y = mnist.train.next_batch(100)
			sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
			# Test the Train Model
			matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

		acc = tf.reduce_mean(tf.cast(matches,tf.float32))
		print("Accuracy : ")
		print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))

if __name__ == '__main__':
	main()
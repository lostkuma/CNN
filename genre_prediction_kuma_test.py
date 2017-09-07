# Useful links:
#
#  Batching input: https://www.tensorflow.org/programmers_guide/reading_data
#
#  Saving progress: https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops/saving_and_restoring_variables
#
#  Convolutional neural network: https://www.tensorflow.org/tutorials/deep_cnn

import os
import tensorflow as tf
import numpy as np

TRAINING = True # True False

TRAINING_FILENAME = 'train.image.genre.listing.csv'
TESTING_FILENAME = 'test.image.genre.listing.csv'

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_DIRECTORY = 'images_Mar26/'

NUMBER_OF_BATCHES = 2000
NUMBER_OF_IMAGES_PER_BATCH = 128

NUMBER_OF_FILTERS_1ST_CONV = 32
NUMBER_OF_FILTERS_2ND_CONV = 64

LEARNING_RATE = 0.001

CHECKPOINT_DIRECTORY = "checkpoint/"
PREDICTED_OUTPUT_DIRECTORY = "predicted_output/"

# Run using just the CPU:
os.environ["CUDA_VISIBLE_DEVICES"] = ""



def GetInput(filename, number_of_genres):
	# Determine how many types of genres we are predicting by checking
	# how many columns the csv file has:
	csv_default_values = []

	# The first column is the image filename:
	csv_default_values.append(tf.constant([], dtype=tf.string))
		
	for i in range(number_of_genres):
		csv_default_values.append(tf.constant([], dtype=tf.float32))
	
	# Setting up our inputs:
	if TRAINING:
		input_csv_filenames = tf.train.string_input_producer([filename], num_epochs=NUMBER_OF_BATCHES)
	else:
		input_csv_filenames = tf.train.string_input_producer([filename], num_epochs=1)

	unused, input_csv_lines = tf.TextLineReader(skip_header_lines=1).read(input_csv_filenames)

	## List of 44 tensors:
	input_csv_columns = tf.decode_csv(input_csv_lines, record_defaults=csv_default_values)

	input_images_filenames = input_csv_columns[0]
	input_images_filenames = tf.constant(IMAGE_DIRECTORY, dtype=tf.string) + input_images_filenames
	input_images_contents = tf.read_file(input_images_filenames)
	input_images = tf.image.decode_jpeg(input_images_contents, channels=3)
	input_images = tf.image.resize_images(input_images, [IMAGE_WIDTH, IMAGE_HEIGHT])
	input_images.set_shape((IMAGE_WIDTH, IMAGE_HEIGHT, 3))

	input_labels = tf.stack(input_csv_columns[1:])

	input_images_batch, input_labels_batch =  tf.train.batch([input_images, input_labels], batch_size=NUMBER_OF_IMAGES_PER_BATCH)

	return input_images_batch, input_labels_batch


def GetNumOfLabels(filename):
	# get total number of genres (columns-1) from file

	number_of_genres = 0
	with open(filename, 'r') as training_file:
		# Figure out how many columns there are in the data file
		# and add default values to csv_default_values.
		number_of_columns = len(training_file.readline().split(','))
		
		# The first column is the image filename:
		number_of_columns = number_of_columns - 1
		
		number_of_genres = number_of_columns

	return number_of_genres



def CNN(input_images_batch, number_of_genres):
	# Setting up all of the operations to build the layers:

	# Convolutional Layer #1
	# Computes NUMBER_OF_FILTERS_1ST_CONV features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, channels]
	# Output Tensor Shape: [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, NUMBER_OF_FILTERS_1ST_CONV]
	convolution_layer_1 = tf.layers.conv2d(
		inputs=input_images_batch,
		filters=NUMBER_OF_FILTERS_1ST_CONV,
		kernel_size=[7, 7],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, NUMBER_OF_FILTERS_1ST_CONV]
	# Output Tensor Shape: [batch_size, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, NUMBER_OF_FILTERS_1ST_CONV]
	pooling_layer_1 = tf.layers.max_pooling2d(
		inputs=convolution_layer_1,
		pool_size=[2, 2],
		strides=2)
		
	# Convolutional Layer #2
	# Computes NUMBER_OF_FILTERS_2ND_CONV features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, NUMBER_OF_FILTERS_1ST_CONV]
	# Output Tensor Shape: [batch_size, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, NUMBER_OF_FILTERS_2ND_CONV]
	convolution_layer_2 = tf.layers.conv2d(
		inputs=pooling_layer_1,
		filters=NUMBER_OF_FILTERS_2ND_CONV,
		kernel_size=[2, 2],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, NUMBER_OF_FILTERS_2ND_CONV]
	# Output Tensor Shape: [batch_size, IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4, NUMBER_OF_FILTERS_2ND_CONV]
	pooling_layer_2 = tf.layers.max_pooling2d(
		inputs=convolution_layer_2,
		pool_size=[2, 2],
		strides=2)	  
		
	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4, NUMBER_OF_FILTERS_2ND_CONV]
	# Output Tensor Shape: [batch_size, IMAGE_WIDTH / 4 * IMAGE_HEIGHT / 4 * NUMBER_OF_FILTERS_2ND_CONV]

	flattened_size = int((IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4) * NUMBER_OF_FILTERS_2ND_CONV)

	pooling_layer_2_flattened = tf.reshape(pooling_layer_2, [-1, flattened_size])
		 
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, IMAGE_WIDTH / 4 * IMAGE_HEIGHT / 4 * NUMBER_OF_FILTERS_2ND_CONV]
	# Output Tensor Shape: [batch_size, 1024]
	dense_layer = tf.layers.dense(
		inputs=pooling_layer_2_flattened,
		units=1024,
		activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
		  inputs=dense_layer,
		  rate=0.4,
		  training=TRAINING)
		  
	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, number_of_genres]
	logits = tf.layers.dense(inputs=dropout, units=number_of_genres)

	return logits



def main():
	# Check if the check point dir exists
	if not os.path.isdir(CHECKPOINT_DIRECTORY):
		os.mkdir(CHECKPOINT_DIRECTORY)

	if not os.path.isdir(PREDICTED_OUTPUT_DIRECTORY):
		os.mkdir(PREDICTED_OUTPUT_DIRECTORY)

	# get input images depends on mode
	input_filename = None
	if TRAINING:
		input_filename = TRAINING_FILENAME
	else:
		input_filename = TESTING_FILENAME
		
	number_of_genres = GetNumOfLabels(input_filename)

	input_images_batch, input_labels_batch = GetInput(input_filename, number_of_genres)

	logits = CNN(input_images_batch, number_of_genres)

	# Setting up our loss function:
	loss_operation = tf.nn.sigmoid_cross_entropy_with_logits(
		labels=input_labels_batch,
		logits=logits)
		
	loss_operation = tf.reduce_mean(loss_operation)

	# Setting up our training operation using the optimizer:
	train_operation = tf.contrib.layers.optimize_loss(
			loss=loss_operation,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=LEARNING_RATE,
			optimizer="SGD")
			
	prediction_operation = tf.sigmoid(logits, name='predictions')
	
	# get argmax for prediction, return an array of indices
	argmax_operation = tf.argmax(prediction_operation, axis=1)

	# construct a matrix with argmax positions to be 1 and the rest to be 0
	prediction_argmax = tf.one_hot(argmax_operation, depth=number_of_genres)

	# multiple with the original to see which positions remain to be 1
	# the positions remain to be 1 means the prediction is also in the labels
	label_checker = prediction_argmax * input_labels_batch

	# calculate an accuracy based on if top prediction is in the label
	top1_checker_accuracy = tf.reduce_mean(tf.reduce_sum(label_checker, axis=1))

	# calculate norm and normalize predicted and label batches
	#norm_predictions = tf.norm(prediction_operation, axis=1, keep_dims=True)
	#norm_labels = tf.norm(input_labels_batch, axis=1, keep_dims=True)
	
	#predictions_normal = prediction_operation / norm_predictions
	#labels_normal = input_labels_batch / norm_labels
	
	# calculate cosine similarity between predicted and label
	#cosine_similarity_operation = tf.reduce_sum(tf.multiply(predictions_normal, labels_normal), 1)
	#cosine_accuracy_operation = tf.reduce_mean(cosine_similarity_operation)

	# get bool vector for predicted and labels
	#prediction_vector = tf.less_equal(prediction_operation, 0.3)
	#label_vector = tf.less_equal(input_labels_batch, 0.3)

	# check how much of the bool vectors are equal, this returns a bool vector
	#comparison_operation = tf.equal(prediction_vector, label_vector)

	# get the mean of the bool vector to be the accuracy
	#accuracy_operation = tf.reduce_mean(tf.cast(comparison_operation, tf.float32), name='accuracy')

	# Setting up our tensorflow session:
	with tf.Session() as session:
		# Initialize the variables:
		session.run(tf.local_variables_initializer())
		session.run(tf.global_variables_initializer())
		
		# Setting up the coordinator responsible for handing out work:
		coordinator = tf.train.Coordinator()
		
		# Setting up the runner threads that will actually do the work:
		threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

		# Create a saver to save all the variables
		# (only keeps the most recent five)
		saver = tf.train.Saver()
		
		# Setting up the try-catch that will let us know when we have
		# run out of inputs:
		try:
			#print(session.run(input_images_batch))
			#print(session.run(tf.shape(input_images_batch)))
			#print(session.run(input_labels_batch))
			#print(session.run(tf.shape(input_labels_batch)))
			
			if TRAINING:
				step = 0
				output_checker = 0
				while not coordinator.should_stop():
					operations_to_run = [
						train_operation,
						prediction_operation,
						loss_operation,
						top1_checker_accuracy
						#accuracy_operation,
						#cosine_similarity_operation,
						#cosine_accuracy_operation,
					]

					operations_results = session.run(operations_to_run)
					
					unused = operations_results[0]
					predicted = operations_results[1]
					loss = operations_results[2]
					top1_accuracy = operations_results[3]
					#accuracy = operations_results[3]
					#cosine_vec = operations_results[3]
					#cosine = operations_results[4]

					# save
					if step % 10 == 0:
						saver.save(session, CHECKPOINT_DIRECTORY + "model", global_step=step)
						print('Step: ' + str(step) + ' is saved')
						output_checker += 1
						
					if output_checker < 5:
						output_file = PREDICTED_OUTPUT_DIRECTORY + 'recent' + str(output_checker) + '.csv'
						np.savetxt(output_file, predicted, fmt='%.4f', delimiter=' ', newline='\n')
						
					if step % 10 == 0:
						print('Finished step: ' + str(step))
						#print('  predicted: ' + str(predicted))
						print('	 loss: ' + str(loss))
						print('	 top 1 prediction accuracy: ' + str(top1_accuracy))
						#print('	 accuracy: ' + str(accuracy))
						#print('   cosine: \n' + str(cosine_vec))
						#print('   cosine similarity avg: ' + str(cosine))
						
					step += 1
					if output_checker == 5:
						output_checker = 0
			
			# if testing then restor the saved parameters
			else:
				saver.restore(session, CHECKPOINT_DIRECTORY + "model-80")
				
				batch = 0
				while not coordinator.should_stop():
					operations_to_run = [
						prediction_vector,
						label_vector,
						accuracy_operation,
						loss_operation,
					]
					
					operations_results = session.run(operations_to_run)
					
					predicted = operations_results[0]
					actual = operations_results[1]
					loss = operations_results[2]
					accuracy = operations_results[3]
					
					print('Batch: ' + str(batch))
					print('	 loss: ' + str(loss))
					print('	 accuracy: ' + str(accuracy))
					
					batch += 1

		except tf.errors.OutOfRangeError:
			print('Done!')
			
		finally:
			coordinator.request_stop()
			coordinator.join(threads)
		

if __name__ == '__main__':
	main()
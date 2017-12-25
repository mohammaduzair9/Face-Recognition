#solve with freeze graph
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc

import os
import shutil
import sys
import time
import numpy as np
import cv2

import tensorflow as tf

import facenet



# Command line arguments
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', './facenet-export',
                           """Directory where to export the model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
FLAGS = tf.app.flags.FLAGS

model="model-128/model.pb"
model_dir = FLAGS.checkpoint_dir
#class_names=np.load('classnames.npy')
#embeds=np.load('embed.npy').astype(np.float32)
#labels=np.load('labels.npy').astype(np.float32)

def calc_dist(emb):

	nrof_embeds = labels.size
	nrof_images = 1
	dist_array = tf.Variable(tf.zeros([nrof_embeds]),dtype=tf.float32)	
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

        for i in range(nrof_embeds):
        	for j in range(nrof_images):
			sub = tf.subtract(embeds[i,:], emb[j,:])
			sqr = tf.square(sub)
			sums =tf.reduce_sum(sqr)
			sqrt = tf.sqrt(sums)
		
			dist_array[i].assign(sqrt)			
            		
	return dist_array

def main(_):

    with tf.Graph().as_default():
		
      
	with tf.Session() as sess:
		facenet.load_model(model)

		# Retrieve the protobuf graph definition and fix the batch norm nodes
		input_graph_def = sess.graph.as_graph_def()

		# Get input and output tensors
		images_placeholder = sess.graph.get_tensor_by_name("input:0")
		embeddings = sess.graph.get_tensor_by_name("embeddings:0")
		phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
		
		#distances = calc_dist(embeddings)

		# Build the signature_def_map.
		predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(images_placeholder)	
		predict_outputs_tensor_info = tf.saved_model.utils.build_tensor_info(embeddings)				
		#dist_output_tensor_info = tf.saved_model.utils.build_tensor_info(distances)


		prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
		inputs={'images': predict_inputs_tensor_info},
		outputs={'embeds': predict_outputs_tensor_info},
		#'dist': dist_output_tensor_info},
		method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            
		legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

		# Export inference model.
            	output_path = FLAGS.output_dir	
		print('Exporting trained model to', output_path)
		builder = tf.saved_model.builder.SavedModelBuilder(output_path)
            	
		builder.add_meta_graph_and_variables(                
			sess, [tf.saved_model.tag_constants.SERVING],
			signature_def_map={'calculate_embeddings': prediction_signature,},
		 	legacy_init_op=legacy_init_op
		)

		builder.save()
            	print('Successfully exported model to %s' % FLAGS.output_dir)



if __name__ == '__main__':
    tf.app.run()

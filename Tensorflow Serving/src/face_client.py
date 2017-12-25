
'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import time
import os
import cv2
import math
import facenet
import align.detect_face

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# loading embeddings
class_names=np.load('classnames.npy')
embeds=np.load('embed.npy')
labels=np.load('labels.npy')


# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    tt = time.time()
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    #print('Model align load time = {}'.format(time.time()-tt))
    #print(' ')
            
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
	#print("orig_img")
	#print(orig_img)
	#resizing image	
	#(height,width,_) = orig_img.shape
		
	#if width <= height:
	#	ratio = 600/width
	#	new_height = int(ratio * height)
	#	new_width = int(600)
	#else:
	#	ratio = 600/height
	#	new_width = int(ratio * width)
	#	new_height = int(600)

	#img = cv2.resize(orig_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	#print("img")
	#print(img)        
	#img = misc.imresize(orig_img, (600, 600))
        img_size = np.asarray(img.shape)[0:2]
     	#print("img_size")
	#print(img_size)
        tt = time.time()
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        #print('Bounding boxes time = {}'.format(time.time()-tt))
        #print(' ')
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        tt = time.time()
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        #print('Crop and align time = {}'.format(time.time()-tt))
        #print(' ')
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def main(_):
    	host, port = FLAGS.server.split(':')
    	channel = implementations.insecure_channel(host, int(port))
    	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    	
    # Send request
    #with open(FLAGS.image, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
    	#data = f.read()
        request = predict_pb2.PredictRequest()
	images = load_and_align_data(['images/12.jpg'], 160, 44, 0.9)
	
        # Call Facenet model to make prediction on the image
        request.model_spec.name = 'face128'
        request.model_spec.signature_name = 'calculate_embeddings'
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(images, dtype=tf.float32))
	request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False, dtype=tf.bool)) 
	
	start_time=time.time()
        result = stub.Predict(request, 60.0)  # 60 secs timeout
	
	# Convert to friendly python object
        results_dict = {}
        for key in result.outputs:
            tensor_proto = result.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results_dict[key] = nd_array

	#storing embeddings
        emb=results_dict.get("embeddings")	
	feature_time = time.time()-start_time

	# calculate distance between images	
	nrof_embeds = labels.size
        dist_array = np.zeros((nrof_embeds, 1))
	for i in range(nrof_embeds):
        	tmp = embeds[i,:]- emb[0,:]
		sum_squared = np.dot(tmp.T , tmp)
		dist = math.sqrt(sum_squared)		# AVGTIME = 0.09sec
		dist_array[i][0] = dist
        
	# arranging distance in ascending order
	pred_array = dist_array.argmin(0) 
         
	# threshold distance to 0.8
	if dist_array[pred_array[0]][0] < 0.8 :
       		pred_label = labels[pred_array[0]]
       		pred_face = class_names[int(pred_label)]
                
	else : 
       		pred_face = 'Unknown'
            
  	print('Face name:  {}'.format(pred_face))
	print('Face Dist:  {}'.format(dist_array[pred_array[0]][0])) 
        print(' ')
	distance_time = time.time()-start_time-feature_time
		
	print('Feature Calculation time:  {}'.format(feature_time))
	print('Distance Calculation time: {}'.format(distance_time))	
	print('Image Recognition time:    {}'.format(time.time() - start_time))
	print(' ')
      
	


if __name__ == '__main__':
    tf.app.run()

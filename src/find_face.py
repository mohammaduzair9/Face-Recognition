#importing libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import time
import os
import cv2
import facenet
import align.detect_face

def main(sess, graph, target, class_names, labels, embeds):
    
	image_files=[target]
	image_size=160
    	image_margin=44
    
    	with graph.as_default():

        	with sess.as_default():
 			
			
			# Load and align images
            		st = time.time()
            		images, save_img, det= load_and_align_data(image_files, image_size, image_margin, 0.9)
			print('Load and Align Images time = {}'.format(time.time() - st))
            		print(' ')
			
            		# Get input and output tensors
            		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]
			
			nrof_images=len(images)			
			###print(nrof_images)
			emb_array = np.zeros((nrof_images, embedding_size))
            		# Run forward pass to calculate embeddings
			feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            		st = time.time()
            		emb_array = sess.run(embeddings, feed_dict=feed_dict)
            		print('Feature Extraction time = {}'.format(time.time() - st))            
            		print(' ')
 
			# calculate distance between images	
			st = time.time() 
           		nrof_embeds = labels.size
            		dist_array = np.zeros((nrof_embeds, nrof_images))
			pred_face = np.empty(nrof_images, dtype=object)
			
			for i in range(nrof_embeds):
                		for j in range(nrof_images):
					dist = np.sqrt(np.sum(np.square(np.subtract(embeds[i,:], emb_array[j,:]))))
                    			dist_array[i][j] = dist
            		print('Distance Calculation time = {}'.format(time.time() - st))
            		print(' ')

			# arranging distance in ascending order
			pred_array = dist_array.argmin(0) 
			print(pred_array)
         		
			# threshold distance to 0.8
			for i in range(pred_array.size):
            			if dist_array[pred_array[i]][i] < 0.8 :
                			pred_label = labels[pred_array[i]]
                			pred_face[i] = class_names[int(pred_label)]
                		else : 
                			pred_face[i] = 'Unknown'
            				print('Face identified as:')
				face_boundings=det[i]
				face_x1=int(face_boundings[0])
				face_y1=int(face_boundings[1])
				face_x2=int(face_boundings[2])
				face_y2=int(face_boundings[3])
            			
				print(pred_face[i])
           			print(' ')
            			print('Face Distance:')
            			print(dist_array[pred_array[i]][i])            
            			print(' ')
            
	return pred_face
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

	minsize = 40 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	det_arr = []
	det_arr_new = []

	tt = time.time()
	print('Creating networks and loading parameters')
	with tf.Graph().as_default():
        	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        	with sess.as_default():
            		pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    	print('Model align load time = {}'.format(time.time()-tt))
    	print(' ')
            
    	nrof_samples = len(image_paths)
   	img_list = [None] * nrof_samples
    	for i in range(nrof_samples):
        	img = misc.imread(os.path.expanduser(image_paths[i]))
		img_size = np.asarray(img.shape)[0:2]
        	tt = time.time()
        	bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        	print('Bounding boxes time = {}'.format(time.time()-tt))
        	print(' ')
        
		nrof_faces = bounding_boxes.shape[0]
		print(nrof_faces)

		if nrof_faces>0:
			save_img=img.copy()
			det = bounding_boxes[:,0:4]
			img_size = np.asarray(img.shape)[0:2]
			for i in range(nrof_faces):
				face_boundings=det[i]
				det_arr.append(np.squeeze(face_boundings))
								
				face_x1=int(face_boundings[0])
				face_y1=int(face_boundings[1])
				face_x2=int(face_boundings[2])
				face_y2=int(face_boundings[3])
				
				
			for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-margin/2, 0)
                                bb[1] = np.maximum(det[1]-margin/2, 0)
                                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                                
				cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        			aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        			prewhitened = facenet.prewhiten(aligned)
        			det_arr_new.append(prewhitened)
	
	print (det_arr)				
	images = np.stack(det_arr_new)
	return images, save_img, det_arr

if __name__ == '__main__':
    main(sess, graph, target, class_names, labels, embeds)

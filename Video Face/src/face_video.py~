#importing libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imutils.video import WebcamVideoStream
from imutils.video import FPS
import align.detect_face
import tensorflow as tf
from scipy import misc
import numpy as np
import imutils
import facenet
import time
import sys
import cv2
import os
import skvideo.io

minsize = 40 # minimum size of face
threshold = [ 0.7, 0.8, 0.8 ]  # three steps's threshold
factor = 0.709 # scale factor
det_arr = []
det_arr_new = []
nrof_samples = 1
img_list = [None] * nrof_samples
margin =0
detect =1


def main():
        	
	graph = tf.Graph() 
	with graph.as_default():
		
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)        	
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        	with sess.as_default():
			
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
			#stream = cv2.VideoCapture(0)
			stream = skvideo.io.VideoCapture("test.mp4")
			print(stream.isOpened())
			fps = FPS().start()		

			while fps._numFrames < 300:
    				# Capture frame-by-frame
    				(grabbed, frame) = stream.read()
				(height,width,_) = frame.shape
				max_side=400
		
				if width <= height:
					ratio = max_side/width
					new_h = int(ratio * height)
					new_w = int(max_side)
				else:
					ratio = max_side/height
					new_w = int(ratio * width)
					new_h = int(max_side)
		
				frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)				
				
				if(detect==1):
					bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
					nrof_faces = bounding_boxes.shape[0]

					if nrof_faces>0:
						det = bounding_boxes[:,0:4]
						img_size = np.asarray(frame.shape)[0:2]
						for i in range(nrof_faces):
							face_boundings=det[i]
							det_arr.append(np.squeeze(face_boundings))
								
							face_x1=int(np.maximum(face_boundings[0]-margin/2, 0))
							face_y1=int(np.maximum(face_boundings[1]-margin/2, 0))
							face_x2=int(np.minimum(face_boundings[2]+margin/2, img_size[1]))
							face_y2=int(np.minimum(face_boundings[3]+margin/2, img_size[0]))
						
							#face box				
							cv2.rectangle(frame,(face_x1, face_y1), (face_x2, face_y2), ( 255, 0, 0), 2)
					
				cv2.imshow('Video', frame)
				fps.update()
			
				if cv2.waitKey(1) & 0xFF == ord('q'):
        				break

			fps.stop()
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))		
			
			# When everything is done, release the capture
			stream.release()
			cv2.destroyAllWindows()

 
			
if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import time
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn import neighbors 
from sklearn.externals import joblib

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            st = time.time()
            np.random.seed(seed=args.seed)
            
            dataset = facenet.get_dataset(args.data_dir)
            class_names=np.load('classnames.npy')

            # Check that there are at least one training image per class
            dataset = [x for x in dataset if not x.name.replace('_', ' ') in class_names]
            print('Removing already added faces')
            for cls in dataset:
                assert len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset'
            
            if not dataset:
                print('no new faces to be added... terminating')
                exit()             
        
            paths, new_labels = facenet.get_image_paths_and_labels(dataset)
 
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            new_emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                new_emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
            #print(new_emb_array) 
            
            # Train classifier
            new_class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            
            emb_array=np.load('embed.npy')
            labels=np.load('labels.npy')
            

            new_labels = np.array(new_labels)
            new_labels = new_labels + class_names.size
            #print(new_labels)
            
            class_names=np.append(class_names,new_class_names)
            #print(class_names)
            emb_array=np.concatenate((emb_array,new_emb_array), axis=0)
            #print(emb_array)
            labels=np.append(labels,new_labels)
            #print(labels)

            #np.savetxt('embed.txt',emb_array)
            #np.savetxt('labels.txt',labels)
            #np.savetxt('classnames.txt',class_names,fmt='%s')             
            
            np.save('embed',emb_array)
            np.save('labels',labels)
            np.save('classnames',class_names)
            print('Elapsed Time = {}'.format(time.time() - st)) 

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=5)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

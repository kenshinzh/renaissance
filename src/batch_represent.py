'''batch_represent for image folder, only contain 1 level for image'''

#----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Kevin Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#----------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import facenet
import numpy as np
import tensorflow as tf


def main(args):

    # grab all image paths and labels
    print("Finding image paths and targets...\n")
    if args.level == 1:
        image_list = facenet.get_folder_file(args.data_dir)
    elif args.level == 2:
        data = facenet.get_dataset(args.data_dir)
        image_list, label_list = facenet.get_image_paths_and_labels(data)
    else:
        print("the wrong parameter of levels, please choose 1 or 2...\n")

    # create output directory if it doesn't exist
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with tf.Graph().as_default():

		with tf.Session() as sess:

			# load the model
			print("Loading trained model...\n")
			meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.trained_model_dir))
			facenet.load_model(args.trained_model_dir, meta_file, ckpt_file)

		    # Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			image_size = images_placeholder.get_shape()[1]
			embedding_size = embeddings.get_shape()[1]

			# Run forward pass to calculate embeddings
			print('Generating embeddings from images...\n')
			start_time = time.time()
            batch_size = args.batch_size
            nrof_images = len(image_list)
            nrof_batches = int(np.ceil(1.0*nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, embedding_size))

			for i in xrange(nrof_batches):
				start_index = i*batch_size
				end_index = min((i+1)*batch_size, nrof_images)
				paths_batch = image_list[start_index:end_index]
				images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False, image_size=image_size, do_prewhiten=True)
				feed_dict = { images_placeholder:images, phase_train_placeholder:False}
				emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

			time_avg_forward_pass = (time.time() - start_time) / float(nrof_images)
			totaltime =  time.time() - start_time
			print("Forward pass took total time: %.3f[seconds] \n" % totaltime)
			print("Forward pass took avg of %.3f[seconds/image] for %d images\n" % (time_avg_forward_pass, nrof_images))

			print("Finally saving embeddings and gallery to: %s" % (output_dir))
			# save the gallery and embeddings (signatures) as numpy arrays to disk
			# np.save(os.path.join(output_dir, "gallery.npy"), label_list)
			np.save(os.path.join(output_dir, "signatures.npy"), emb_array)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--level', type=int,
        help='level for caculate the image embeddings.', default=1, choices=[1, 2])
    parser.add_argument('data_dir', type=str,
		help='directory of images with structure as seen at the top of this file.')
    parser.add_argument('output_dir', type=str,
		help='directory containing aligned face patches with file structure as seen at the top of this file.')
    parser.add_argument('trained_model_dir', type=str,
        help='Load a trained model before training starts.')
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=50)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
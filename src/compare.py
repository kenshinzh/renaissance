"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align_dlib

def main(args):
    input_images = load_and_align_data_dlib(args.input_dir, args.image_size, args.dlib_face_predictor, center_crop=False)
    target_images = load_data(args.target_dir, args.image_size)

    nrof_input_images = len(os.listdir(os.path.expanduser(args.input_dir)))
    nrof_target_images = len(os.listdir(os.path.expanduser(args.target_dir)))
    images_index = [None] * (nrof_input_images+nrof_target_images)
    for i in range(nrof_input_images+nrof_target_images):
        if i < nrof_input_images:
            images_index[i] = input_images[i]
        else:
            images_index[i] = target_images[i-nrof_input_images]
    images = np.stack(images_index)

    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
#            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
#           print('Metagraph file: %s' % meta_file)
#           print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
    
            # Get input and output tensors
#           images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict_input = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict_input)

            #nrof_input_images = len(os.listdir(os.path.expanduser(args.input_dir)))
            #nrof_target_images = len(os.listdir(os.path.expanduser(args.target_dir)))

            input_dataset = facenet.get_folder(args.input_dir)
            target_dataset = facenet.get_folder(args.target_dir)

            print('Input Images:')
            for i in range(nrof_input_images):
                print('%1d: %s' % (i, os.path.expanduser(input_dataset[i])))
            print('')

            print('Compare Images:')
            for i in range(nrof_input_images, nrof_target_images+nrof_input_images):
                print('%1d: %s' % (i, os.path.expanduser(target_dataset[i-nrof_input_images])))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('Compare Image', end='')
            for i in range(nrof_input_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_input_images, nrof_target_images):
                # print(emb[i, :])
                print('%1d  ' % i, end='')
                for j in range(nrof_input_images):
                    # print(emb[j, :])
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')


def load_data(target_dir, image_size):

    target_dataset = facenet.get_folder(target_dir)
    nrof_samples = len(os.listdir(os.path.expanduser(target_dir)))
    img_list = [None] * nrof_samples

    for i in xrange(nrof_samples):
        print(target_dataset[i])
        img = misc.imread(os.path.expanduser(target_dataset[i]))
        aligned = misc.imresize(img, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def load_and_align_data_dlib(input_dir, image_size, dlib_face_predictor, center_crop=False):

    align = align_dlib.AlignDlib(os.path.expanduser(dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

    minsize = 80  # minimum size of face
    scale = float(minsize) / image_size
    input_dataset = facenet.get_folder(input_dir)
    nrof_samples = len(os.listdir(os.path.expanduser(input_dir)))
    img_list = [None] * nrof_samples

    for i in xrange(nrof_samples):
        print(input_dataset[i])
        img = misc.imread(os.path.expanduser(input_dataset[i]))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        if center_crop:
            scaled = misc.imresize(img, center_crop, interp='bilinear')
            sz1 = int(scaled.shape[1] / 2) # MacOX need int, while Linux don't need
            sz2 = int(image_size / 2)  # MacOX need int, while Linux don't need
            cropped = scaled[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
        else:
            cropped = align.align(image_size, img, landmarkIndices=landmarkIndices,
                                  skipMulti=False, scale=scale)
        if cropped is not None:
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('--dlib_face_predictor', type=str,
        help='File containing the dlib face predictor.', default='../data/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('input_dir', type=str,
        help='Input Directory of Images to compare')
    parser.add_argument('target_dir', type=str,
        help='Target Directory of Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

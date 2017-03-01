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

import argparse
import os
import sys

import align_dlib
import detect_face
import facenet
import numpy as np
import tensorflow as tf
from scipy import misc


def main(args):

    if args.align_method == 'mtcnn':
        input_images = load_and_align_data_mtcnn(args.input_dir, args.image_size, args.margin,
                                                 args.gpu_memory_fraction, args.prealigned_scale)
        target_images = load_and_align_data_mtcnn(args.target_dir,args.image_size, args.margin,
                                                 args.gpu_memory_fraction, args.prealigned_scale)

    else:
        input_images = load_and_align_data_dlib(args.input_dir, args.image_size,
                                                args.dlib_face_predictor, args.prealigned_scale)
        target_images = load_and_align_data_dlib(args.target_dir, args.image_size,
                                                args.dlib_face_predictor, args.prealigned_scale)


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

            # nrof_input_images = len(os.listdir(os.path.expanduser(args.input_dir)))
            # nrof_target_images = len(os.listdir(os.path.expanduser(args.target_dir)))

            input_dataset = facenet.get_folder_file(args.input_dir)
            target_dataset = facenet.get_folder_file(args.target_dir)

            print('Input Images:')
            for i in range(nrof_input_images):
                print('%1d: %s' % (i, os.path.expanduser(input_dataset[i])))
            print('')

            print('Compare Images:')
            for i in range(nrof_input_images, nrof_target_images + nrof_input_images):
                print('%1d: %s' % (i, os.path.expanduser(target_dataset[i - nrof_input_images])))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('Compare Image', end='')
            for i in range(nrof_input_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_input_images, nrof_target_images + nrof_input_images):
                # print(emb[i, :]) Target_image_embeddings
                print('%1d       ' % i, end='')
                for j in range(nrof_input_images):
                    # print(emb[j, :]) Input_Image_embeddings
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')



def load_and_align_data_mtcnn(dir, image_size, margin, gpu_memory_fraction, prealigned_scale):

    minsize = 80  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, '../data/')

    dataset = facenet.get_folder_file(dir)
    nrof_samples = len((os.listdir(os.path.expanduser(dir))))
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(dataset[i]))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det = det[index, :]
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            # plt.imshow(prewhitened)
            img_list[i] = prewhitened
        else:
            print('Unable to align "%s"' % dataset[i])
            scaled = misc.imresize(img, prealigned_scale, interp='bilinear')
            sz1 = int(scaled.shape[1] / 2)
            sz2 = int(image_size / 2)
            cropped = scaled[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            # plt.imshow(prewhitened)
            img_list[i] = prewhitened
    images = np.stack(img_list)
    return images



def load_and_align_data_dlib(dir, image_size, dlib_face_predictor, prealigned_scale):

    align = align_dlib.AlignDlib(os.path.expanduser(dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

    minsize = 80  # minimum size of face
    scale = float(minsize) / image_size
    dataset = facenet.get_folder_file(dir)
    nrof_samples = len(os.listdir(os.path.expanduser(dir)))
    img_list = [None] * nrof_samples

    for i in xrange(nrof_samples):
        # print(input_dataset[i])
        img = misc.imread(os.path.expanduser(dataset[i]))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        dets = align.getAllFaceBoundingBoxes(img)
        nrof_faces = len(dets)
        if nrof_faces > 0:
            cropped = align.align(image_size, img, landmarkIndices=landmarkIndices,
                                  skipMulti=False, scale=scale)
        else:
            scaled = misc.imresize(img, prealigned_scale, interp='bilinear')
            sz1 = int(scaled.shape[1] / 2)
            sz2 = int(image_size / 2)
            cropped = scaled[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
        #if center_crop:
         #   scaled = misc.imresize(img, center_crop, interp='bilinear')
          #  sz1 = int(scaled.shape[1] / 2) # MacOX need int, while Linux don't need
           # sz2 = int(image_size / 2)  # MacOX need int, while Linux don't need
            # cropped = scaled[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
        #else:
        if cropped is not None:
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            #plt.imshow(prewhitened)
            #filename = os.path.splitext(os.path.split(input_dataset[i])[1])[0]
            #output_filename = os.path.join(test_input_folders, filename + '.png')
            #misc.imsave(output_filename, prewhitened)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('input_dir', type=str,
        help='Input Directory of Images to compare')
    parser.add_argument('target_dir', type=str,
        help='Target Directory of Images to compare')
    parser.add_argument('--dlib_face_predictor', type=str,
        help='File containing the dlib face predictor.', default='../data/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--align_method', type=str, choices=['mtcnn', 'dlib'],
        help='Face align and load algorithm to use', default='dlib')
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--prealigned_scale', type=float,
        help='The amount of scaling to apply to prealigned images before taking the center crop.', default=0.87)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
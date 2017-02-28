"""Performs face alignment and stores face thumbnails in the output directory."""

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
import sys
import os
import argparse
import numpy as np
import shutil
import align_dlib
import facenet

def main(args):
    align = align_dlib.AlignDlib(os.path.expanduser(args.dlib_face_predictor))
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE
    output_dir = os.path.expanduser(args.output_dir)
    input_dir = os.path.expanduser(args.input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    # src_path,_ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    folder = facenet.get_folder(input_dir)
    # Scale the image such that the face fills the frame when cropped to crop_size
    scale = float(args.face_size) / args.image_size

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        nrof_images = len(os.listdir(os.path.expanduser(input_dir)))
        for i in range(nrof_images):
            output_folder_dir = os.path.join(output_dir, os.path.splitext(os.path.split(folder[i])[1])[0])
            if not os.path.exists(output_folder_dir):
                os.makedirs(output_folder_dir)
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(folder[i])[1])[0]
            output_filename = os.path.join(output_folder_dir, filename)
            print(output_filename)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(folder[i])
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(folder[i], e)
                    text_file.write('%s \n' % (errorMessage))
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        text_file.write('Unable to align %s\n' % (folder[i]))
                        shutil.rmtree(output_folder_dir)
                        print('Unable to align "%s"' % (folder[i]))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]

                    dets = align.getAllFaceBoundingBoxes(img)
                    nrof_faces = len(dets)
                    if nrof_faces > 0:
                        for j in range(nrof_faces):
                            bb = dets[j]
                            aligned = align.align(args.image_size, img, bb, landmarkIndices=landmarkIndices,
                                              skipMulti=False, scale=scale)
                            filename = os.path.splitext(os.path.split(folder[i])[1])[0]
                            output_filename = os.path.join(output_folder_dir, filename + '_' + str(j).zfill(4) + '.png')
                            print(folder[i])
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, aligned)
                            print(bb)
                            text_file.write('%s %s \n' % (output_filename, bb))

                        # aligned = align.align(args.image_size, img, landmarkIndices=landmarkIndices,
                        #                      skipMulti=False, scale=scale)
                    else:
                        text_file.write('Unable to align %s\n' % (folder[i]))
                        shutil.rmtree(output_folder_dir)
                        print('Unable to align "%s"' % folder[i])
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--dlib_face_predictor', type=str,
        help='File containing the dlib face predictor.', default='../data/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--face_size', type=int,
        help='Size of the face thumbnail (height, width) in pixels.', default=160)


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

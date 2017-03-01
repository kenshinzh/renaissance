"""Download the VGG face dataset from URLs given by http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz
"""
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

import os
import socket
from httplib import HTTPException
from urllib2 import HTTPError, URLError

import numpy as np
from scipy import misc
from skimage import io

socket.setdefaulttimeout(30)
dataset_dir = './vgg_face'
RESULT_ROOT = './vgg_face_datasets'
if not os.path.exists(RESULT_ROOT):
    os.mkdir(RESULT_ROOT)
output_format = 'png'
image_size = 256
steps = 500

def download((names, urls, bboxes)):
    """
        download from urls into folder names using wget
    """
    assert (len(names) == len(urls))
    assert (len(names) == len(bboxes))

    # download using external wget
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(names)):
        directory = os.path.join(RESULT_ROOT, names[i])
        if not os.path.exists(directory):
            os.mkdir(directory)
        fname = names[i] + '_' + str(i).zfill(4) + '.jpg'
        errname = names[i] + '_' + str(i).zfill(4) + '.err'
        image_path = os.path.join(directory, fname)
        error_path = os.path.join(directory, errname)
        print ("downloading", image_path)
        if os.path.exists(image_path):
            print ("already downloaded, skipping...")
            continue
        else:
            if not os.path.exists(image_path) and not os.path.exists(error_path):
                try:
                    img = io.imread(urls[i], mode='RGB')
                except (HTTPException, HTTPError, URLError, IOError, ValueError, IndexError, OSError) as e:
                    err_message = '{}: {}'.format(urls[i], e)
                    save_error_message_file(error_path, err_message)
                else:
                    try:
                        if img.ndim == 2:
                            img = to_rgb(img)
                        if img.ndim != 3:
                            raise ValueError('Wrong number of image dimensions')
                        hist = np.histogram(img, 255, density=True)
                        if hist[0][0]>0.9 and hist[0][254]>0.9:
                            raise ValueError('Image is mainly black or white')
                        else:
                            # Crop image according to dataset descriptor
                            img_cropped = img[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2]), :]
                            # Scale to 256x256
                            img_resized = misc.imresize(img_cropped, (image_size,image_size))
                            # Save image as .png
                            misc.imsave(image_path, img_resized)
                    except ValueError as e:
                        error_message = '{}: {}'.format(url, e)
                        save_error_message_file(error_path, error_message)

def save_error_message_file(filename, error_message):
    print(error_message)
    with open(filename, "w") as textfile:
        textfile.write(error_message)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


if __name__ == '__main__':

    combine_filename = os.path.join(RESULT_ROOT, 'vgg_face_datasets.txt')

    with open(combine_filename, "w") as text_file:
        get_files = os.listdir(dataset_dir)
        for get_file in get_files:
            if get_file.endswith('.txt'):
                with open(os.path.join(dataset_dir, get_file), 'r') as fd:
                    class_name = os.path.splitext(get_file)[0]
                    index = 1
                    url = ""
                    bbox = []
                    for line in fd.readlines():
                        components = line.split(' ')
                        assert (len(components) == 9)
                        url = components[1]
                        bbox = np.rint(np.array(map(float, components[2:6])))
                        text_file.write('%s %05d %s %d,%d,%d,%d \n' % (class_name, index, url, bbox[0], bbox[1], bbox[2], bbox[3]))
                        index += 1


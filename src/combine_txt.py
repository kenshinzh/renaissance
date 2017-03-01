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
import numpy as np

dataset_dir = './vgg_face'
RESULT_ROOT = './vgg_face_datasets'
if not os.path.exists(RESULT_ROOT):
    os.mkdir(RESULT_ROOT)

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
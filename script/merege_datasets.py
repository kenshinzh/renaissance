"""merger datasets from vgg_datasets, youtube_faces and facescrub
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
import sys
import argparse
import shutil
import numpy as np


# FOLDERS = '~/Documents/Facenet/vgg/vgg_datasets'

def main(args):
    dst_root = os.path.expanduser(args.dst_datasets)
    if not os.path.isdir(dst_root):  # Create the log directory if it doesn't exist
        os.makedirs(dst_root)
    log_file = os.path.join(dst_root, 'logfile.txt')
    src_lists = args.dataset_dir
    nrof_success = 0
    nrof_exists = 0
    with open(bounding_boxes_filename, "w") as log_file:
       for i in range(len(src_lists)):
          if os.path.isdir(os.path.expanduser(src_lists[i])):
            src_exp = os.path.expanduser(src_lists[i])
            if os.path.isdir(src_exp):
                src_lists = os.listdir(src_exp)
                src_lists.sort()
                for j in range(len(src_lists)):
                    src_dir = os.path.join(src_exp, src_lists[j])
                    dst_dir = os.path.join(dst_root, src_lists[j])
                    if (not os.path.exists(dst_dir)) and (not os.path.isfile(src_lists[j])):
                        shutil.move(src_dir, dst_dir)
                        nrof_success +=1
                    else:
                        log_file.write('%s and %s are the same name please double check\n' % (src_dir, dst_dir))
                        print('%s already exists, please double check' % dst_dir)
                        nrof_exists +=1
          else:
            print('%s is not a directory, please choose the datasets folder.' % os.path.expanduser(src_lists[i]))

    print('Total success moved %i folders to the %s.' % (nrof_success, dst_root))
    print('Total %i folders already existed in %s, please double check.' % (nrof_exists, dst_root))


def move(src_dir, dst_dir):
    src_exp = os.path.expanduser(src_dir)
    dst_exp = os.path.expanduser(dst_dir)

    if os.path.isdir(src_exp):
        src_lists = os.listdir(src_exp)
        src_lists.sort()
        for i in range(len(src_lists)):
            if (not os.path.exists(os.path.join(dst_exp,src_lists[i]))) and (not os.path.isfile(src_lists[i])):
                shutil.move(os.path.join(src_exp, src_lists[i]), os.path.join(dst_exp, src_lists[i]))
            else:
                print('%s already exists, please double check' % os.path.join(dst_exp, src_lists[i]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dir', type=str, nargs='+',
                        help='Directory of datasets needs to be merged')
    parser.add_argument('--dst_datasets', type=str,
                        help='Directory that to merger the datasets.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
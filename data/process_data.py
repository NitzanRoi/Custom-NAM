# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import numpy as np
import scipy.misc as misc

dataset_name = sys.argv[1]
im_size=64

if (len(sys.argv) == 3):
    im_size = sys.argv[2] # for 32*32 case. other dimensions are not supported

fn = "%s/train" % dataset_name
ls = os.listdir(fn)
n = len(ls)
x = np.zeros((n, im_size, im_size, 3), dtype="uint8")
y = np.zeros((n, im_size, im_size, 3), dtype="uint8")

for i in range(n):
    im = misc.imread("%s/train/%s" % (dataset_name, ls[i]))
    im = misc.imresize(im, (im_size, im_size*2))
    x[i] = im[:, im_size:]
    y[i] = im[:, :im_size]

np.save("%s_x" % dataset_name, x)
np.save("%s_y" % dataset_name, y)


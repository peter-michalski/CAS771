from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export
import scipy.io

mymat = scipy.io.loadmat('test_images_var0_36.mat')
mymattrain = scipy.io.loadmat('train_images_var0_36.mat')


@keras_export('keras.datasets.mnist.load_data')

def load_data(path='mnist.npz'):
  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  path = get_file(
      path,
      origin=origin_folder + 'mnist.npz',
      file_hash=
      '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
  with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    mymat_data = mymat['test_images_var0_36']
    mymattrain_data = mymattrain["train_images_var0_36"]
    x_train = mymattrain_data
    x_test = mymat_data
    return (x_train, y_train), (x_test, y_test)
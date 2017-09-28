import glob
import os
from random import shuffle
from sys import platform

import numpy as np
import tensorflow as tf
from PIL import Image


SIZE = 256

if platform == 'win32':
    FACES_PATH = 'D:/Temp/PublicMM1/matlab/input crop'
    FEATURES_PATH = 'D:/Temp/PublicMM1/matlab/labels'
    TF_RECORD_PATH = 'D:/Temp/TfRecords'
    CLEAR = 'cls'
elif platform == 'linux':
    FACES_PATH = '/home/jscholz/Code/pic23d/data/input crop'
    FEATURES_PATH = '/home/jscholz/Code/pic23d/data/labels'
    TF_RECORD_PATH = '/home/jscholz/Code/pic23d/data/TfRecords'
    CLEAR = 'clear'



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_generator(path):
    addrs = glob.glob(path)
    shuffle(addrs)
    for i, path in enumerate(addrs):
        fname = os.path.basename(path)
        label_path = f'{FEATURES_PATH}/{fname[:-6]}.label'
        try:
            im = Image.open(path)
            if im.height != SIZE or im.width != SIZE:
                if im.height > im.width:
                    im.thumbnail((SIZE, 999999), Image.BICUBIC)
                else:
                    im.thumbnail((999999, SIZE), Image.BICUBIC)
                im = im.crop(((im.width - SIZE) / 2,
                              (im.height - SIZE) / 2,
                              (im.width + SIZE) / 2,
                              (im.height + SIZE) / 2))
                # im.show()
            # img = np.array(im)/SIZE
            img = np.array(im)
            label = np.loadtxt(label_path)
            label = list(np.reshape(label, (2 * 199)))
            if img.shape == (SIZE, SIZE, 3):
                # Black & White Pictures are discarded
                yield (img, label)
        except Exception as e:
            print(e)


def create_tfrecord():

    max_files = 5000
    path = f'{FACES_PATH}/*.jpg'

    print(f'getting Data from: {path}')
    features = feature_generator(path)
    writer = None
    num_files = len(glob.glob(path))
    print(f'found {num_files} files in {FACES_PATH}/')
    for i, feat in enumerate(features):
        if not i % 50:
            os.system(CLEAR)
            print(f'writing: {TF_RECORD_PATH}/Dataset_{SIZE}_{i//max_files:06d}.tfrecords')
            print(f'data: {i}/{num_files}')
        if not i % max_files:
            if writer is not None:
                writer.close()
            writer = tf.python_io.TFRecordWriter(
                f'{TF_RECORD_PATH}/Dataset_{SIZE}_{i//max_files:06d}.tfrecords')

        img, label = feat

        feature = {'label': _float_feature(label),
                   'image': _bytes_feature(img.tostring())}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()

create_tfrecord()

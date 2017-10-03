import os
import glob
from sys import platform
from threading import Thread
from pic23d_model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)


if platform == 'win32':
    DEV = '/gpu:0'
    BASE_PATH = 'D:/Temp'
elif platform == 'linux':
    DEV = '/gpu:0'
    BASE_PATH = '/home/jscholz/Code/pic23d'


def train_thread(params, epochs):
    mod = Model(**params)
    print(mod)
    with mod.start_session() as sess:
        # mod.restore(sess, f'{BASE_PATH}/cnnlog/afn=relu,bs=200,lr=5e-03/')
        mod.train(epochs)
        # for img in glob.glob(f'{BASE_PATH}/*.jpg'):
        # print(img)
        # mod.predict(img)

    mod = None


def main():

    params = {'batchsize': 200,
              'learn_rate': 0.002,
              'activation_func': tf.nn.crelu,
              'device': DEV,
              'summary': 'verbose',
              'conv_layers': 8,
              'parameter_index': None,
              'optimizer': tf.train.AdamOptimizer,
              'comment': f'weightedMSEloss_{64}',
              'pic_size': 64}

    mod = Model(**params)
    with mod.start_session() as sess:

        mod.restore(sess, f'{BASE_PATH}/cnnlog/afn=crelu,bs=200,lr=1e-03,layer=16,op=AdamOptimizer,comment=weightedMSEloss_64x64,run=2')
        for img in glob.glob(f'{BASE_PATH}/examples/*.jpg'):
            mod.predict(img)

if __name__ == '__main__':
    main()

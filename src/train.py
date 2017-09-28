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
elif platform == 'linux':
    DEV = '/gpu:0'


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
    batchsizes = [200]
    learning_rates = [0.001 / (10**x) for x in range(2)]
    activation_funcs = [tf.nn.crelu]
    layers = [4, 8, 16]
    epochs = 30000

    for bs in batchsizes:
        for lr in learning_rates:
            for ly in layers:
                for afn in activation_funcs:
                    params = {'batchsize': bs,
                              'learn_rate': lr,
                              'activation_func': afn,
                              'device': DEV,
                              'summary': 'verbose',
                              'conv_layers': ly,
                              'parameter_index': None,
                              'optimizer': tf.train.AdamOptimizer,
                              'comment': 'weightedMSEloss_64x64',
                              'pic_size': 64}

                    t = Thread(target=train_thread, args=(params, epochs))
                    t.start()
                    t.join()
                    #mod = model(**params)
                    #with mod.start_session() as sess:
                    #    mod.restore(sess, f'{BASE_PATH}/cnnlog/afn=crelu,bs=200,lr=1e-03,layer=4,op=AdamOptimizer,comment=huber_allParameters_64x64,run=0')
                    #    mod.predict(f'{BASE_PATH}/*.jpg')

if __name__ == '__main__':
    main()

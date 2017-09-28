import glob
import os
from sys import platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.DEBUG)

if platform == 'win32':
    FACES_PATH = 'D:/Temp/PublicMM1/matlab/input crop'
    FEATURES_PATH = 'D:/Temp/PublicMM1/matlab/labels'
    TF_RECORD_PATH = f'D:/Temp/TfRecords'
    BASE_PATH = 'D:/Temp'
    CLEAR = 'cls'
    DEV = '/gpu:0'
elif platform == 'linux':
    FACES_PATH = '/home/jscholz/Code/pic23d/data/input crop'
    FEATURES_PATH = '/home/jscholz/Code/pic23d/data/labels'
    TF_RECORD_PATH = f'/home/jscholz/Code/pic23d/data/TfRecords'
    BASE_PATH = '/home/jscholz/Code/pic23d'
    CLEAR = 'clear'
    DEV = '/gpu:0'


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.15), name='weight')


def new_biases(length):
    return tf.Variable(tf.constant(0.0, shape=[length]), name='bias')


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def scaled_tanh(features, name=None):
    return 1.7159 * tf.nn.tanh((2 / 3) * features, name)


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.

    print(input.get_shape())
    print(weights.get_shape())

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer = tf.nn.bias_add(layer, biases)

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


class Model:
    def __init__(self, batchsize=100,
                 learn_rate=0.001,
                 activation_func=tf.nn.relu,
                 device='/gpu:0',
                 summary='less',
                 conv_layers=100,
                 parameter_index=None,
                 optimizer=tf.train.GradientDescentOptimizer,
                 comment='',
                 pic_size=64):
        tf.reset_default_graph()
        self.parameter_index = parameter_index
        self.batchsize = batchsize
        self.learn_rate = learn_rate
        self.activation_func = activation_func
        self.dev = device
        self.summary = summary
        self.conv_layers = conv_layers if not conv_layers % 2 else conv_layers - 1
        self.optimizer = optimizer
        self.size=pic_size
        path = f'{BASE_PATH}/cnnlog/afn={activation_func.__name__},' \
               f'bs={batchsize},' \
               f'lr={learn_rate:.0e},' \
               f'layer={self.conv_layers},' \
               f'op={self.optimizer.__name__},' \
               f'comment={comment}'
               #f'param_ids={self.parameter_index}-{self.parameter_index+10},' \
        self.session_path = f'{path},run={len(glob.glob(path+"*"))}'
        self.convs = []
        self.grads = []
        self.imgs, self.lbls = self.build_queue()

        with tf.device(self.dev):
            self.input_placeholder, self.loss = self.build_model(self.imgs, optimizer)
            self.graph = tf.get_default_graph()
            self.saver = tf.train.Saver()
            self.create_summarys()

        self.merged = tf.summary.merge_all()

    def read_and_decode(self, fname_queue, batch_size):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(fname_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([199, 2], tf.float32)
            })

        image = tf.decode_raw(features['image'], tf.uint8)

        image_shape = [self.size, self.size, 3]
        label_shape = [199, 2]

        image = tf.reshape(image, image_shape)
        label = tf.reshape(features['label'],
                           label_shape) / 1  # Labels haben sigma = 1 als verteilung Skallierung bleibt wie sie ist
        if self.parameter_index is not None:
            label = label[self.parameter_index:self.parameter_index + 10,
                    0]  # Nur die ersten beiden Featurepunkte der Form
        image = tf.cast(image, tf.float32) * (1 / 255) - 0.5

        min_after_dequeue = batch_size * 4
        capacity = min_after_dequeue + 400 * batch_size

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=2,
                                                min_after_dequeue=min_after_dequeue)
        return images, labels

    def tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def operation(self, name):
        return self.graph.get_operation_by_name(name)

    def start_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.graph.as_default()
        self.session = tf.Session(config=config, graph=self.graph)
        return self.session

    def build_queue(self):
        with tf.name_scope('Data_queues'):
            filename_queue1 = tf.train.string_input_producer(
                glob.glob(f'{TF_RECORD_PATH}_{self.size}/*.tfrecords'))

        return self.read_and_decode(filename_queue1, self.batchsize)

    def create_summarys(self):

        with tf.name_scope('scalars'):
            # tf.summary.scalar('accuracy', self.tensor('accuracy/Mean:0'))
            tf.summary.scalar('errror', self.loss)

            metrics = [self.tensor('dnn/loss/moments/Squeeze_1:0'),
                       self.tensor('dnn/loss/mean:0'),
                       self.tensor('dnn/loss/max:0'),
                       self.tensor('dnn/loss/min:0'),
                       tf.reduce_mean(tf.squared_difference(self.lbls, self.tensor('dnn/prediction:0')))]

            names = ['var', 'mean', 'max', 'min', 'mse']

            for m, n in zip(metrics, names):
                tf.summary.scalar(f'{n}', tf.reduce_mean(m))

        if self.summary == 'verbose':
            with tf.name_scope('images'):
                tf.summary.image(self.imgs.name, self.imgs, max_outputs=1)
                for c in self.convs:
                    for i in range(4):
                        tf.summary.image(c.name, tf.expand_dims(c[:, :, :, i], axis=3), max_outputs=1)

            with tf.name_scope('biases_weights'):
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
            with tf.name_scope('gradients'):
                for grad, var in self.grads: \
                        tf.summary.histogram(var.name + '/gradient', grad)
            with tf.name_scope('metrics'):
                for m, n in zip(metrics, names):
                    tf.summary.histogram(f'{n}', m)

    def build_model(self, data_input, optimizer_func):
        with tf.name_scope('dnn'):
            input_placeholder = tf.placeholder_with_default(data_input, shape=data_input.shape, name="input")
            convs = []
            outer_convs = 1
            filters = 16

            out = input_placeholder
            for i in range(self.conv_layers):
                out_shape = out.get_shape().as_list()
                print(out_shape[3])
                layer_conv, weights_conv = \
                    new_conv_layer(input=out,
                                   num_input_channels=out_shape[3],
                                   filter_size=3,
                                   num_filters=16,
                                   use_pooling=not (i + 1) % (self.conv_layers // 2))
                out = layer_conv
                self.convs.append(layer_conv)

            # out = tf.stack([x for x in convs], axis=4)
            print(out)

            flat, _ = flatten_layer(out)
            # pool3_flat = tf.reshape(out, [-1, out.num_elements() / self.batchsize]) #[-1, filters * outer_convs * 64 * 256])

            dense1 = tf.layers.dense(inputs=flat,
                                     units=filters * outer_convs * 64,
                                     activation=scaled_tanh,
                                     use_bias=True,
                                     name='pre_logits')

            dropout = tf.layers.dropout(  # Dropout ist erstmal draussen
                inputs=dense1,
                rate=0.4,
                training=True)

            dense2 = tf.layers.dense(inputs=dense1,
                                     units=398,
                                     activation=None,
                                     use_bias=True,
                                     name='logits')

            logits = tf.reshape(dense2, [-1, 199, 2], name='prediction')

            print(logits)

            with tf.name_scope('loss'):
                # tf.losses.log_loss(logits + 1e-7, self.lbls)

                # Uses default weight of 1.0

                _, var_pred = tf.nn.moments(logits, axes=[0])
                _, var_truth = tf.nn.moments(self.lbls, axes=[0])

                mean_pred = tf.reduce_mean(logits, axis=0, name='mean')
                max_pred = tf.reduce_max(logits, axis=0, name='max')
                min_pred = tf.reduce_min(logits, axis=0, name='min')

                shape_tex_Ev = tf.reshape(tf.convert_to_tensor([[1.0000, 0.6286, 0.4939, 0.3542, 0.3124, 0.2364, 0.2179, 0.2048, 0.2010, 0.1828, 0.1722, 0.1620, 0.1466, 0.1300, 0.1203, 0.1148, 0.1117, 0.1079, 0.1007, 0.0990, 0.0930, 0.0860, 0.0849, 0.0831, 0.0793, 0.0787, 0.0767, 0.0738, 0.0682, 0.0653, 0.0643, 0.0606, 0.0588, 0.0570, 0.0560, 0.0538, 0.0525, 0.0517, 0.0494, 0.0484, 0.0464, 0.0448, 0.0426, 0.0417, 0.0404, 0.0400, 0.0393, 0.0386, 0.0374, 0.0366, 0.0357, 0.0355, 0.0349, 0.0333, 0.0330, 0.0327, 0.0319, 0.0305, 0.0302, 0.0294, 0.0291, 0.0289, 0.0281, 0.0268, 0.0260, 0.0257, 0.0254, 0.0250, 0.0246, 0.0243, 0.0240, 0.0236, 0.0230, 0.0225, 0.0223, 0.0221, 0.0216, 0.0211, 0.0209, 0.0206, 0.0203, 0.0199, 0.0196, 0.0193, 0.0188, 0.0187, 0.0179, 0.0178, 0.0177, 0.0173, 0.0172, 0.0169, 0.0168, 0.0166, 0.0164, 0.0161, 0.0156, 0.0154, 0.0153, 0.0152, 0.0150, 0.0149, 0.0147, 0.0144, 0.0141, 0.0139, 0.0137, 0.0135, 0.0134, 0.0131, 0.0131, 0.0129, 0.0127, 0.0124, 0.0123, 0.0122, 0.0121, 0.0119, 0.0117, 0.0115, 0.0115, 0.0113, 0.0111, 0.0110, 0.0107, 0.0106, 0.0105, 0.0105, 0.0104, 0.0101, 0.0101, 0.0099, 0.0098, 0.0097, 0.0097, 0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0090, 0.0090, 0.0089, 0.0088, 0.0087, 0.0086, 0.0085, 0.0084, 0.0083, 0.0082, 0.0079, 0.0078, 0.0077, 0.0077, 0.0076, 0.0075, 0.0074, 0.0074, 0.0072, 0.0072, 0.0071, 0.0070, 0.0070, 0.0069, 0.0068, 0.0067, 0.0067, 0.0066, 0.0065, 0.0064, 0.0063, 0.0063, 0.0062, 0.0061, 0.0060, 0.0059, 0.0058, 0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0055, 0.0054, 0.0053, 0.0053, 0.0053, 0.0052, 0.0051, 0.0051, 0.0049, 0.0048, 0.0048, 0.0047, 0.0047, 0.0046, 0.0044, 0.0044, 0.0042],[1.0000, 0.4933, 0.4092, 0.3445, 0.3208, 0.3006, 0.2468, 0.2245, 0.2142, 0.1976, 0.1930, 0.1883, 0.1730, 0.1634, 0.1598, 0.1522, 0.1504, 0.1439, 0.1395, 0.1327, 0.1301, 0.1261, 0.1225, 0.1194, 0.1183, 0.1150, 0.1137, 0.1108, 0.1076, 0.1068, 0.1039, 0.1016, 0.1005, 0.0981, 0.0967, 0.0958, 0.0937, 0.0926, 0.0917, 0.0898, 0.0869, 0.0866, 0.0862, 0.0845, 0.0838, 0.0832, 0.0822, 0.0812, 0.0809, 0.0795, 0.0783, 0.0772, 0.0769, 0.0760, 0.0756, 0.0753, 0.0741, 0.0732, 0.0722, 0.0716, 0.0710, 0.0701, 0.0695, 0.0691, 0.0686, 0.0680, 0.0668, 0.0664, 0.0662, 0.0652, 0.0649, 0.0646, 0.0639, 0.0634, 0.0630, 0.0628, 0.0620, 0.0616, 0.0611, 0.0606, 0.0602, 0.0596, 0.0595, 0.0587, 0.0583, 0.0580, 0.0577, 0.0574, 0.0568, 0.0563, 0.0558, 0.0556, 0.0552, 0.0551, 0.0547, 0.0545, 0.0542, 0.0537, 0.0537, 0.0532, 0.0530, 0.0528, 0.0524, 0.0522, 0.0519, 0.0514, 0.0512, 0.0509, 0.0505, 0.0503, 0.0500, 0.0496, 0.0494, 0.0491, 0.0485, 0.0484, 0.0482, 0.0481, 0.0478, 0.0474, 0.0472, 0.0469, 0.0468, 0.0465, 0.0464, 0.0461, 0.0459, 0.0457, 0.0454, 0.0454, 0.0453, 0.0449, 0.0446, 0.0443, 0.0442, 0.0439, 0.0436, 0.0435, 0.0432, 0.0431, 0.0428, 0.0427, 0.0424, 0.0422, 0.0422, 0.0420, 0.0416, 0.0416, 0.0412, 0.0410, 0.0408, 0.0408, 0.0406, 0.0403, 0.0401, 0.0399, 0.0398, 0.0396, 0.0392, 0.0390, 0.0389, 0.0387, 0.0386, 0.0385, 0.0383, 0.0381, 0.0379, 0.0377, 0.0375, 0.0374, 0.0373, 0.0370, 0.0369, 0.0367, 0.0366, 0.0364, 0.0363, 0.0359, 0.0356, 0.0355, 0.0354, 0.0352, 0.0351, 0.0349, 0.0346, 0.0345, 0.0344, 0.0341, 0.0338, 0.0337, 0.0335, 0.0333, 0.0330, 0.0328, 0.0324, 0.0322, 0.0320, 0.0316, 0.0310]]), [199, 2])

                error = tf.reduce_mean(tf.squared_difference(self.lbls, logits) * shape_tex_Ev)
                #error = tf.losses.mean_squared_error(labels=self.lbls, predictions=logits, weights=shape_tex_Ev)
                #print(shape_tex_Ev)


                #error = tf.losses.huber_loss(logits, self.lbls)
                # error = tf.reduce_sum(logits * tf.log(logits/(self.lbls + 1e-12))) # KL Divergence

                # error = tf.reduce_mean(-tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=logits/self.lbls)) # KL Divergence
                # error = tf.reduce_mean(tf.losses.absolute_difference(labels=self.lbls, predictions=logits))
                # error = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.lbls, predictions=logits))

                # tf.losses.mean_squared_error(var_pred, var_truth)

                # All the losses are collected into the `GraphKeys.LOSSES` collection.
                # error = tf.get_collection(tf.GraphKeys.LOSSES)

                # error = tf.losses.compute_weighted_loss(error)
                # print(error)
                # error = tf.contrib.losses.mean_squared_error(logits,
                #                                             self.lbls)
                # error = -(tf.losses.cosine_distance(logits, self.lbls, dim=0)-1)
                # error = -tf.reduce_sum(logits * tf.log(tf.clip_by_value(self.lbls, 1e-10, 1.0)))
                # error = tf.losses.log_loss(logits,
                #                           self.lbls)
                # mean_squared_error = tf.reduce_mean(error, name='mean_squared_error')

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype='int64')
            #learning_rate = tf.train.exponential_decay(self.learn_rate, global_step, 10000, 0.95, staircase=True)
            optimizer = optimizer_func(learning_rate=self.learn_rate)
            print(global_step)

            # Op to calculate every variable gradient
            grads = tf.gradients(error, tf.trainable_variables())
            self.grads = list(zip(grads, tf.trainable_variables()))
            # Op to update all variables according to their gradient
            optimizer.apply_gradients(grads_and_vars=self.grads, global_step=global_step)
            train_op = optimizer.minimize(loss=error, global_step=global_step, name='train_op')

        # with tf.name_scope('correct_prediction'):
        #    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.lbls, 1))
        # with tf.name_scope('accuracy'):
        #    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return input_placeholder, error

    def __repr__(self):
        return f'CNN Test Model Object: Batchsize:{self.batchsize},' \
               f'Learnrate:{self.learn_rate},' \
               f'Activationfunction:{self.activation_func.__name__},' \
               f'Covolutional Layers:{self.conv_layers},' \
               f'optimizer:{self.optimizer.__name__},' #\
               #f'param_ids:{self.parameter_index}:{self.parameter_index+2}'

    def predict(self, image):
        # tf.reset_default_graph()
        im = Image.open(image)
        if im.height != self.size or im.width != self.size:
            if im.height > im.width:
                im.thumbnail((self.size, 999999), Image.BICUBIC)
            else:
                im.thumbnail((999999, self.size), Image.BICUBIC)
            im = im.crop(((im.width - self.size) / 2,
                          (im.height - self.size) / 2,
                          (im.width + self.size) / 2,
                          (im.height + self.size) / 2))

        im = np.asarray(im, dtype=np.uint8) * (1 / 255) - 0.5
        im = np.expand_dims(im, axis=0)

        im_input = np.zeros((self.batchsize, 256, 256, 3))
        im_input[:im.shape[0], :im.shape[1], :im.shape[2], :im.shape[3]] = im

        pred = self.session.run(self.tensor('dnn/prediction:0'), feed_dict={self.input_placeholder: im_input})
        print(f'{pred}')

    def train(self, epochs):

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        self.session.run(init_op)

        writer = tf.summary.FileWriter(f'{self.session_path}', self.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = self.tensor('train/global_step:0').eval()
        epochs = epochs + step
        loss = []
        try:
            while step < epochs:

                _, step, los = self.session.run([self.operation('train/train_op'),
                                                 self.tensor('train/global_step:0'),
                                                 self.loss])

                loss.append(los)
                print(f'Epoch:{step}, loss:{los:.6f}', end="\r")

                if not step % 500 or step == epochs:
                    print('saving model')
                    self.saver.save(self.session,
                                    f'{self.session_path}/model-{step:06d}')
                if not step % 50:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                    run_metadata = tf.RunMetadata()

                    operations = [self.merged,
                                  self.tensor('dnn/prediction:0'),
                                  self.lbls,
                                  tf.reduce_mean(self.tensor('dnn/loss/moments/Squeeze_1:0')),
                                  tf.reduce_mean(self.tensor('dnn/loss/mean:0')),
                                  tf.reduce_mean(self.tensor('dnn/loss/max:0')),
                                  tf.reduce_mean(self.tensor('dnn/loss/min:0'))]

                    mrg, pred, label, var, mean, max_, min_ = self.session.run(operations,
                                                                               options=run_options,
                                                                               run_metadata=run_metadata)
                    # os.system(CLEAR)
                    writer.add_run_metadata(run_metadata, f'step:{step}')

                    print(f'mean loss: {np.mean(loss)}')
                    #for p, l in zip(pred, label):
                    #   print(p)
                    #   print(l)
                    #   print(abs(p-l))
                    #   print('------------------')
                    print(f'mean: {mean}')
                    print(f'var: {var}')
                    print(f'min: {min_}')
                    print(f'max: {max_}')
                    loss = []
                    writer.add_summary(mrg, step)
                    # print(
                    #    f'Epoch:{step},'
                    #    f'Batchsize:{self.batchsize}, '
                    #    f'activation_func:{self.activation_func.__name__}, '
                    #    f'lr:{self.learn_rate}')
                    # print(f'loss:{los:.6f}')
                    # print(f'acc:{acc:.2f}')
                    writer.flush()

        except tf.errors.OutOfRangeError as e:
            print(e)

        finally:
            writer.close()
            coord.request_stop()
            coord.join(threads)

    def restore(self, session, path=None):
        if path is not None:
            used_path = path
        else:
            used_path = self.session_path

        models = [x for x in glob.glob(f'{used_path}/model*') if '.meta' in x]

        if len(models) > 0:
            self.saver = tf.train.import_meta_graph(max(models))

        self.saver.restore(session, tf.train.latest_checkpoint(used_path))
        self.graph = tf.get_default_graph()

        step = self.graph.get_tensor_by_name('train/global_step:0')

        print(f'restored model {tf.train.latest_checkpoint(used_path)} Epoch: {step.eval(session=session)}')

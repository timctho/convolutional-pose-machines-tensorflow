import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim


class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0

    def build_model(self, input_image, center_map, batch_size):
        self.batch_size = batch_size
        self.input_image = input_image
        self.center_map = center_map
        with tf.variable_scope('pooled_center_map'):
            # center map is a gaussion template which gather the respose
            self.center_map = slim.avg_pool2d(self.center_map,
                                              [9, 9], stride=8,
                                              padding='SAME',
                                              scope='center_map')

        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('sub_stages'):
                net = slim.conv2d(input_image, 64, [3, 3], scope='sub_conv1')
                net = slim.conv2d(net, 64, [3, 3], scope='sub_conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv5')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv6')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv7')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv8')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv9')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv10')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv11')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv12')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv13')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv14')

                self.sub_stage_img_feature = slim.conv2d(net, 128, [3, 3],
                                                         scope='sub_stage_img_feature')

            with tf.variable_scope('stage_1'):
                conv1 = slim.conv2d(self.sub_stage_img_feature, 512, [1, 1],
                                    scope='conv1')
                self.stage_heatmap.append(slim.conv2d(conv1, self.joints, [1, 1],
                                                      scope='stage_heatmap'))

            for stage in range(2, self.stages + 1):
                self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage-2],
                                                 self.sub_stage_img_feature,
                                                 # self.center_map,
                                                 ],
                                                axis=3)
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer()):
                mid_net = slim.conv2d(self.current_featuremap, 128, [7, 7], scope='mid_conv1')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv2')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv3')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv4')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid_conv5')
                mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='mid_conv6')
                self.current_heatmap = slim.conv2d(mid_net, self.joints, [1, 1],
                                                   scope='mid_conv7')
                self.stage_heatmap.append(self.current_heatmap)

    def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
        self.gt_heatmap = gt_heatmap
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                       name='l2_loss') / self.batch_size
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer='Adam')
        self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        # weight_file_object = open(weight_file_path, 'rb')
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # conv1
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/biases')

                loaded_kernel = weights['conv1_' + str(layer)]
                loaded_bias = weights['conv1_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv2
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/biases')

                loaded_kernel = weights['conv2_' + str(layer)]
                loaded_bias = weights['conv2_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv3
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/biases')

                loaded_kernel = weights['conv3_' + str(layer)]
                loaded_bias = weights['conv3_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/biases')

                loaded_kernel = weights['conv4_' + str(layer)]
                loaded_bias = weights['conv4_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 12) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 12) + '/biases')

                loaded_kernel = weights['conv5_' + str(layer)]
                loaded_bias = weights['conv5_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/weights')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/biases')

            loaded_kernel = weights['conv5_3_CPM']
            loaded_bias = weights['conv5_3_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            ## stage 1
            conv_kernel = tf.get_variable('stage_1/conv1/weights')
            conv_bias = tf.get_variable('stage_1/conv1/biases')

            loaded_kernel = weights['conv6_1_CPM']
            loaded_bias = weights['conv6_1_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                conv_kernel = tf.get_variable('stage_1/stage_heatmap/weights')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/biases')

                loaded_kernel = weights['conv6_2_CPM']
                loaded_bias = weights['conv6_2_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/weights')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/biases')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))

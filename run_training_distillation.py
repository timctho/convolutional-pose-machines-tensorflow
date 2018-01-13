import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import models.nets.cpm_hand as teacher_model
import models.nets.cpm_hand_v2 as student_model
from utils import cpm_utils, utils
import Ensemble_data_generator
from config import FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Teacher(object):
    def __init__(self, input_size, output_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = teacher_model.CPM_Model(input_size, output_size, 6, 21, img_type='RGB', is_training=False)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.sess = tf.Session(config=sess_config, graph=self.graph)
            self._init_vars()
            self.saver = tf.train.Saver()

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    @property
    def all_graph_nodes(self):
        with self.graph.as_default() as graph:
            return [n.name for n in graph.as_graph_def().node]


class Student(object):
    def __init__(self, input_size, output_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = student_model.CPM_Model(input_size, output_size, 3, 21, img_type='RGB', is_training=True)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.sess = tf.Session(config=sess_config, graph=self.graph)
            self._init_vars()
            self.saver = tf.train.Saver()

    def _init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    @property
    def all_graph_nodes(self):
        with self.graph.as_default() as graph:
            return [n.name for n in graph.as_graph_def().node]


def guided_loss(student_stages_output_tensor, teacher_stages_output_tensor):
    if len(student_stages_output_tensor) != len(teacher_stages_output_tensor):
        raise ValueError('Length must be equal between teacher and student nodes')

    batch_size = tf.cast(tf.shape(student_stages_output_tensor[0])[0], dtype=tf.float32)

    stages = len(student_stages_output_tensor)
    stage_loss = [0 for _ in range(stages)]
    total_loss = 0
    for stage in range(stages):
        with tf.variable_scope('stage_' + str(stage + 1) + '_loss'):
            stage_loss[stage] = tf.nn.l2_loss(student_stages_output_tensor[stage] -
                                              teacher_stages_output_tensor[stage], name='L2_loss') / batch_size
            tf.summary.scalar('stage_' + str(stage + 1) + '_loss', stage_loss[stage])
        with tf.variable_scope('total_loss'):
            total_loss += stage_loss[stage]
    tf.summary.scalar('total loss', total_loss)
    return total_loss, stage_loss


def gt_loss(student_stages_output_tensor, gt_output):
    batch_size = tf.cast(tf.shape(student_stages_output_tensor[0])[0], dtype=tf.float32)

    stages = len(student_stages_output_tensor)
    stage_loss_gt = [0 for _ in range(stages)]
    total_loss_gt = 0
    for stage in range(stages):
        with tf.variable_scope('stage_' + str(stage + 1) + '_gt_loss'):
            stage_loss_gt[stage] = tf.nn.l2_loss(student_stages_output_tensor[stage] -
                                                 gt_output, name='L2_loss') / batch_size
            tf.summary.scalar('stage_' + str(stage + 1) + '_gt_loss', stage_loss_gt[stage])
        with tf.variable_scope('total_gt_loss'):
            total_loss_gt += stage_loss_gt[stage]
    tf.summary.scalar('gt loss', total_loss_gt)
    return total_loss_gt, stage_loss_gt


def get_train_op(total_loss, init_lr, lr_decay_rate, lr_decay_step, optimizer):
    with tf.variable_scope('train'):
        global_step = tc.framework.get_or_create_global_step()

        cur_lr = tf.train.exponential_decay(init_lr,
                                            global_step=global_step,
                                            decay_steps=lr_decay_step,
                                            decay_rate=lr_decay_rate
                                            )
        tf.summary.scalar('learning rate', cur_lr)
        train_op = tf.contrib.layers.optimize_loss(loss=total_loss,
                                                   global_step=global_step,
                                                   learning_rate=cur_lr,
                                                   optimizer=optimizer)
        return {'train_op': train_op, 'global_step': global_step, 'cur_lr': cur_lr}


def print_current_training_stats(global_step, cur_lr, stage_losses, total_loss, stage_gt_losses, total_gt_loss,
                                 time_elapsed):
    nStages = len(stage_losses)
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGS.training_iters,
                                                                                 cur_lr, time_elapsed)
    losses = ' | '.join(
        ['S{} loss: {:>7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in range(nStages)])
    losses += ' | Total loss: {}'.format(total_loss)
    gt_losses = ' | '.join(
        ['S{} gt_loss: {:>7.2f}'.format(stage_num + 1, stage_gt_losses[stage_num]) for stage_num in
         range(len(stage_gt_losses))])
    gt_losses += ' | Total gt_loss: {}'.format(total_gt_loss)
    print(stats)
    print(losses)
    print(gt_losses + '\n')


def train():
    LEARN_TARGET = 'full_tensor'  # [full_tensor, response]

    guide_node_names = [  # 'sub_stages/sub_stage_img_feature/BiasAdd:0',
        'stage_2/mid_conv7/BiasAdd:0',
        'stage_4/mid_conv7/BiasAdd:0',
        'stage_6/mid_conv7/BiasAdd:0']
    studied_node_names = [  # 'sub_stages/sub_stage_img_feature/BiasAdd:0',
        'stage_1/stage_heatmap/BiasAdd:0',
        'stage_2/mid_conv7/BiasAdd:0',
        'stage_3/mid_conv7/BiasAdd:0']

    teacher_input_size = 368
    student_input_size = 128
    down_sample = 4
    alpha = 0.0
    resize_scale = float(student_input_size) / teacher_input_size
    teacher = Teacher(teacher_input_size, teacher_input_size // down_sample)
    student = Student(student_input_size, student_input_size // down_sample)

    g = Ensemble_data_generator.ensemble_data_generator(FLAGS.train_img_dir,
                                                        FLAGS.bg_img_dir,
                                                        FLAGS.batch_size, 368, True, True,
                                                        FLAGS.augmentation_config, FLAGS.hnm, FLAGS.do_cropping)

    # Get teacher middle network output nodes
    guide_nodes = []
    with teacher.graph.as_default() as cur_graph:
        if LEARN_TARGET == 'full_tensor':
            for name in guide_node_names:
                node = cur_graph.get_tensor_by_name(name=name)
                guide_nodes.append(node)
        elif LEARN_TARGET == 'response':
            for name in guide_node_names:
                node = cur_graph.get_tensor_by_name(name=name)
                node = tf.abs(node)
                node = tf.reduce_mean(node, axis=3)
                guide_nodes.append(node)

    studied_nodes = []
    with student.graph.as_default() as cur_graph:
        if LEARN_TARGET == 'full_tensor':
            for name in studied_node_names:
                node = cur_graph.get_tensor_by_name(name=name)
                studied_nodes.append(node)
        elif LEARN_TARGET == 'response':
            for name in studied_node_names:
                node = cur_graph.get_tensor_by_name(name=name)
                node = tf.abs(node)
                node = tf.reduce_mean(node, axis=3)
                studied_nodes.append(node)

    with student.graph.as_default():
        teacher_output_list = [tf.placeholder(dtype=tf.float32,
                                              shape=[None, 32, 32, 22],
                                              name='guided_output_{}'.format(i))
                               for i in range(len(guide_nodes))]
        student_loss, student_stage_loss = guided_loss(studied_nodes, teacher_output_list)
        gt_loss_part, gt_stage_loss_part = gt_loss(student.model.stage_heatmap, student.model.gt_hmap_placeholder)
        total_loss = student_loss + alpha * gt_loss_part

        train_ops_vars = get_train_op(total_loss, init_lr=FLAGS.init_lr,
                                      lr_decay_rate=FLAGS.lr_decay_rate,
                                      lr_decay_step=FLAGS.lr_decay_step,
                                      optimizer='RMSProp')

    with student.graph.as_default():
        for v in tf.global_variables():
            print('in student', v.name)
        student.sess.run(tf.global_variables_initializer())
        # student.saver.restore(student.sess, 'guided_cpm-10000')

        # gs = student.graph.get_tensor_by_name('train/global_step:0')
        # student.sess.run(tf.assign(gs, value=10000))
        # print(student.sess.run(gs))

    teacher.saver.restore(teacher.sess, 'cpm_hand')

    train_iters = 300000
    for train_iter in range(train_iters):
        t1 = time.time()

        # Size 368
        batch_imgs_large, batch_joints_large = g.next()

        # Size 128
        batch_size = batch_imgs_large.shape[0]
        batch_imgs_small = np.zeros(shape=(batch_size, student_input_size, student_input_size, 3))
        batch_joints_small = np.zeros(shape=(batch_size, 21, 2))
        for i in range(batch_size):
            batch_imgs_small[i] = cv2.resize(batch_imgs_large[i], (student_input_size, student_input_size))
            batch_joints_small[i] = batch_joints_large[i] * resize_scale
        # Generate heatmaps from joints
        hm_size = student_input_size / down_sample
        batch_gt_heatmap_np = cpm_utils.make_heatmaps_from_joints_openpose(student_input_size,
                                                                           hm_size,
                                                                           FLAGS.joint_gaussian_variance,
                                                                           batch_joints_small)

        # Normalize
        batch_imgs_large = batch_imgs_large / 255.0 - 0.5
        batch_imgs_small = batch_imgs_small / 255.0 - 0.5

        teacher_output_heatmaps, \
        guide_output = teacher.sess.run([teacher.model.stage_heatmap,
                                         guide_nodes],
                                        feed_dict={teacher.model.input_images: batch_imgs_large})

        teacher_resized_heatmaps = [
            np.zeros(shape=(batch_size, student_input_size / down_sample, student_input_size / down_sample, 22))
            for _ in range(len(guide_output))]
        for i in range(len(guide_output)):
            for batch_num in range(batch_size):
                teacher_resized_heatmaps[i][batch_num] = cv2.resize(guide_output[i][batch_num], (
                student_input_size / down_sample, student_input_size / down_sample))

        feed_dict = {student.model.input_images: batch_imgs_small,
                     student.model.gt_hmap_placeholder: batch_gt_heatmap_np}
        for k, v in zip(teacher_output_list, teacher_resized_heatmaps):
            feed_dict.update({k: v})

        student_output_heatmaps, \
        loss, \
        stage_loss_np, \
        gt_loss_np, \
        stage_gt_loss_np, \
        _, \
        global_step_np, \
        cur_lr_np = student.sess.run([student.model.stage_heatmap,
                                      student_loss,
                                      student_stage_loss,
                                      gt_loss_part,
                                      gt_stage_loss_part,
                                      train_ops_vars['train_op'],
                                      train_ops_vars['global_step'],
                                      train_ops_vars['cur_lr']],
                                     feed_dict=feed_dict)

        if (train_iter + 1) % 10 == 0:
            color_img = (batch_imgs_large[0] + 0.5) * 255.0
            hm_img_teacher = utils.draw_stages_heatmaps(teacher_output_heatmaps, student_input_size)
            hm_img_student = utils.draw_stages_heatmaps(student_output_heatmaps, student_input_size)
            cv2.imshow('hm teacher', hm_img_teacher)
            cv2.imshow('hm student', hm_img_student)
            cv2.imshow('color', color_img.astype(np.uint8))
            cv2.waitKey(10)

        if (train_iter + 1) % 10000 == 0:
            with student.graph.as_default():
                student.saver.save(sess=student.sess, save_path='distillation', global_step=global_step_np)

        print_current_training_stats(global_step_np, cur_lr_np, stage_loss_np, loss, stage_gt_loss_np, gt_loss_np,
                                     time.time() - t1)


if __name__ == '__main__':
    train()

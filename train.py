import tensorflow as tf
import numpy as np
import cv2


from utils import cpm_utils, tf_utils
from models.nets import cpm_hand_slim, cpm_body_slim


"""Hyper Parameters
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfr_data_files',
                           default_value=['./train_128x128.tfrecords'],
                           docstring='Training data tfrecords')
tf.app.flags.DEFINE_string('pretrained_model',
                           default_value='cpm_body.pkl',
                           docstring='Pretrained mode')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=128,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('heatmap_size',
                            default_value=32,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('center_radius',
                            default_value=21,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('num_of_joints',
                            default_value=14,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('batch_size',
                            default_value=32,
                            docstring='Training mini-batch size')
tf.app.flags.DEFINE_integer('training_iterations',
                            default_value=300000,
                            docstring='Training iterations')
tf.app.flags.DEFINE_integer('lr',
                            default_value=0.0001,
                            docstring='Learning rate')
tf.app.flags.DEFINE_integer('lr_decay_rate',
                            default_value=0.5,
                            docstring='Learning rate decay rate')
tf.app.flags.DEFINE_integer('lr_decay_step',
                            default_value=20000,
                            docstring='Learning rate decay steps')
tf.app.flags.DEFINE_string('saved_model_name',
                           default_value='_cpm_body_i' + str(FLAGS.input_size) + 'x' + str(
                               FLAGS.input_size) + '_o' + str(FLAGS.heatmap_size) + \
                                         'x' + str(FLAGS.heatmap_size) + '_' + str(
                               FLAGS.stages) + 's',
                           docstring='Saved model name')
tf.app.flags.DEFINE_string('log_file_name',
                           default_value='_cpm_body_i' + str(FLAGS.input_size) + 'x' + str(
                               FLAGS.input_size) + '_o' + str(FLAGS.heatmap_size) + \
                                         'x' + str(FLAGS.heatmap_size) + '_' + str(
                               FLAGS.stages) + 's',
                           docstring='Log file name')
tf.app.flags.DEFINE_string('log_dir',
                           default_value='logs/_cpm_body_i' + str(FLAGS.input_size) + 'x' + str(
                               FLAGS.input_size) + '_o' + str(FLAGS.heatmap_size) + \
                                         'x' + str(FLAGS.heatmap_size) + '_' + str(
                               FLAGS.stages) + 's',
                           docstring='Log directory name')
tf.app.flags.DEFINE_string('color_channel',
                           default_value='RGB',
                           docstring='')


def main(argv):
    """Build graph
    """
    batch_x, batch_c, batch_y, batch_x_orig = tf_utils.read_batch_cpm(FLAGS.tfr_data_files, FLAGS.input_size,
                                                                      FLAGS.heatmap_size, FLAGS.num_of_joints,
                                                                      FLAGS.center_radius, FLAGS.batch_size)
    if FLAGS.color_channel == 'RGB':
        input_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3),
                                           name='input_placeholer')
    elif FLAGS.color_channel == 'GRAY':
        input_placeholder = tf.placeholder(dtype=tf.float32,
                                           shape=(FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1),
                                           name='input_placeholer')
    cmap_placeholder = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 1),
                                      name='cmap_placeholder')
    hmap_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=(FLAGS.batch_size,
                                             FLAGS.heatmap_size,
                                             FLAGS.heatmap_size,
                                             FLAGS.num_of_joints + 1),
                                      name='hmap_placeholder')

    model = cpm_body_slim.CPM_Model(FLAGS.stages, FLAGS.num_of_joints + 1)
    model.build_model(input_placeholder, cmap_placeholder, FLAGS.batch_size)
    model.build_loss(hmap_placeholder, FLAGS.lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_step)
    print('=====Model Build=====\n')

    """Training
    """
    with tf.Session() as sess:

        # Create dataset queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ## Create summary
        tf_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph, filename_suffix=FLAGS.log_file_name)

        ## Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore weights
        if FLAGS.pretrained_model is not None:
            if FLAGS.pretrained_model.endswith('.pkl'):
                model.load_weights_from_file(FLAGS.pretrained_model, sess, finetune=True)

                # Check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

            else:
                saver.restore(sess, FLAGS.pretrained_model)

                # check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

        while True:

            # Read in batch data
            batch_x_np, batch_y_np, batch_c_np = sess.run([batch_x,
                                                           batch_y,
                                                           batch_c])

            # Warp training images
            for img_num in range(batch_x_np.shape[0]):
                deg1 = (2 * np.random.rand() - 1) * 50
                deg2 = (2 * np.random.rand() - 1) * 50
                batch_x_np[img_num, ...] = cpm_utils.warpImage(batch_x_np[img_num, ...],
                                                           0, deg1, deg2, 1, 30)
                batch_y_np[img_num, ...] = cpm_utils.warpImage(batch_y_np[img_num, ...],
                                                           0, deg1, deg2, 1, 30)
                batch_y_np[img_num, :, :, FLAGS.num_of_joints] = np.ones(shape=(FLAGS.input_size, FLAGS.input_size)) - \
                                                                 np.max(
                                                                     batch_y_np[img_num, :, :, 0:FLAGS.num_of_joints],
                                                                     axis=2)
                batch_c_np[img_num, ...] = cpm_utils.warpImage(batch_c_np[img_num, ...],
                                                           0, deg1, deg2, 1, 30).reshape(
                    (FLAGS.input_size, FLAGS.input_size, 1))

            # Convert image to grayscale
            if FLAGS.color_channel == 'GRAY':
                batch_x_gray_np = np.zeros((batch_x_np.shape[0], FLAGS.input_size, FLAGS.input_size, 1))
                for img_num in range(batch_x_np.shape[0]):
                    tmp = batch_x_np[img_num, ...]
                    tmp += 0.5
                    tmp *= 255
                    tmp = np.dot(tmp[..., :3], [0.114, 0.587, 0.299])
                    tmp /= 255
                    tmp -= 0.5
                    batch_x_gray_np[img_num, ...] = tmp.reshape((FLAGS.input_size, FLAGS.input_size, 1))
                batch_x_np = batch_x_gray_np

            # Recreate heatmaps
            gt_heatmap_np = cpm_utils.make_gaussian_batch(batch_y_np, FLAGS.heatmap_size, 3)

            # Update once
            stage_losses_np, total_loss_np, _, summary, current_lr, \
            stage_heatmap_np, global_step = sess.run([model.stage_loss,
                                                      model.total_loss,
                                                      model.train_op,
                                                      model.merged_summary,
                                                      model.lr,
                                                      model.stage_heatmap,
                                                      model.global_step
                                                      ],
                                                     feed_dict={input_placeholder: batch_x_np,
                                                                cmap_placeholder: batch_c_np,
                                                                hmap_placeholder: gt_heatmap_np})

            # Write logs
            tf_writer.add_summary(summary, global_step)

            # Draw intermediate results
            if global_step % 50 == 0:

                if FLAGS.color_channel == 'GRAY':
                    demo_img = np.repeat(batch_x_np[0], 3, axis=2)
                    demo_img += 0.5
                elif FLAGS.color_channel == 'RGB':
                    demo_img = batch_x_np[0] + 0.5
                demo_stage_heatmaps = []
                for stage in range(FLAGS.stages):
                    demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                    demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
                    demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
                    demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                    demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                    demo_stage_heatmaps.append(demo_stage_heatmap)

                demo_gt_heatmap = gt_heatmap_np[0, :, :, 0:FLAGS.num_of_joints].reshape(
                    (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                demo_gt_heatmap = cv2.resize(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size))
                demo_gt_heatmap = np.amax(demo_gt_heatmap, axis=2)
                demo_gt_heatmap = np.reshape(demo_gt_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                demo_gt_heatmap = np.repeat(demo_gt_heatmap, 3, axis=2)

                if FLAGS.stages > 4:
                    upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]),
                                               axis=1)
                    blend_img = 0.5 * demo_gt_heatmap + 0.5 * demo_img
                    lower_img = np.concatenate((demo_stage_heatmaps[FLAGS.stages - 1], demo_gt_heatmap, blend_img),
                                               axis=1)
                    demo_img = np.concatenate((upper_img, lower_img), axis=0)
                    cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)
                else:
                    upper_img = np.concatenate((demo_stage_heatmaps[FLAGS.stages - 1], demo_gt_heatmap, demo_img),
                                               axis=1)
                    cv2.imshow('current heatmap', (upper_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)

            print('##========Iter {:>6d}========##'.format(global_step))
            print('Current learning rate: {:.8f}'.format(current_lr))
            for stage_num in range(FLAGS.stages):
                print('Stage {} loss: {:>.3f}'.format(stage_num + 1, stage_losses_np[stage_num]))
            print('Total loss: {:>.3f}\n\n'.format(total_loss_np))

            # Save models
            if global_step % 5000 == 1:
                save_path_str = 'models/' + FLAGS.saved_model_name
                saver.save(sess=sess, save_path=save_path_str, global_step=global_step)
                print('\nModel checkpoint saved...\n')

            # Finish training
            if global_step == FLAGS.training_iterations:
                break

        coord.request_stop()
        coord.join(threads)

    print('Training done.')


if __name__ == '__main__':
    tf.app.run()
